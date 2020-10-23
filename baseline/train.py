from argparse import ArgumentError
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from torchvision.models.resnet import resnet50


import l5kit
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer

from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory

import os
from contextlib import nullcontext

import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt

import models
from models import BaselineModel

import loss_functions

def plot_progress(losses, save=False):
    plt.plot([x[1] for x in losses], [x[0] for x in losses])
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.yscale('log')
    plt.grid(True)
    
    if save:
        plt.savefig("./loss_graphs/loss_iter_" + str(losses[-1][1]) + ".png")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, help="Where to find the competition data")
    parser.add_argument("--config", type=str, help="Location of the config file")
    parser.add_argument("--weight-file", type=str, default=None, help="path tho the file containing the weights for the model")
    parser.add_argument("--multi-gpu", action='store_true', help="Enables training on multiple GPU's. Defaults to false")

    args = parser.parse_args()
    print("multi-gpu =", args.multi_gpu)

    # Load config
    os.environ["L5KIT_DATA_FOLDER"] = args.input_dir
    dm = LocalDataManager(None)
    cfg = load_config_data(args.config)

    multi_mode = cfg["model_params"]["multi_mode"]
    
    plot_every_n_steps = cfg['plot_every_n_steps']
    
    # Create dataloaders
    train_cfg = cfg["train_data_loader"]
    rasterizer = build_rasterizer(cfg, dm)

    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
    train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                                  num_workers=train_cfg["num_workers"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create model
    model = BaselineModel(cfg)
    if args.weight_file != None:
        model.load_state_dict(
            torch.load(args.weight_file)
        )
    
    model.to(device)
    multi_gpu = args.multi_gpu
    if multi_gpu:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    loss_config = cfg["train_params"]["loss"]

    criterion = None

    if multi_mode:
        if loss_config == "mse":
            criterion = loss_functions.multi_mode_mse
        elif loss_config == "log_likelihood":
            criterion = loss_functions.pytorch_neg_multi_log_likelihood_batch
    elif loss_config == "mse":
        criterion = nn.MSELoss(reduction="none")
    
    if criterion == None:
        raise ArgumentError("unknown loss function in config: " + loss_config)

    # Train loop
    print("Start train loop")
    
    tr_it = iter(train_dataloader)
    max_steps = cfg["train_params"]["max_num_steps"]

    if max_steps >= 0:
        progress_bar = tqdm(range(max_steps))
        all_data = False
    else:
        progress_bar = tqdm(tr_it)
        all_data = True
        iteration_index = 0
    
    losses_train = []
    losses_plot = []
    
    
    model.train()
    torch.set_grad_enabled(True)

    
    for itr in progress_bar:
        if all_data:
            data = itr
            iteration_index += 1
        else:
            try:
                data = next(tr_it)
            except StopIteration:
                tr_it = iter(train_dataloader)
                data = next(tr_it)
            iteration_index = itr + 1

        # Calculate loss
        targets = data["target_positions"].to(device)
        target_availabilities = data["target_availabilities"].to(device)

        with autocast() if not torch.cuda.is_available() else nullcontext():
            if multi_mode:
                predictions, confidences = model(
                    data["image"].to(device)
                )

                loss = loss_functions.pytorch_neg_multi_log_likelihood_batch(targets, predictions, confidences, target_availabilities)
            else:
                target_availabilities = target_availabilities.unsqueeze(-1)
                output = model(
                    data["image"].to(device)
                ).reshape(targets.shape)

                loss = criterion(output, targets)

                loss = loss * target_availabilities
                loss = loss.mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iteration_index) % cfg['train_params']['checkpoint_every_n_steps'] == 0 and not cfg['debug']:
            if multi_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, f'model_state_{iteration_index}.pth')
            
        if (iteration_index) % plot_every_n_steps == 0 and not cfg['debug']:
            losses_plot.append((np.mean(losses_train), iteration_index))
            plot_progress(losses_plot, save=True)
        
        losses_train.append(loss.item())
        losses_train = losses_train[-plot_every_n_steps:]
        progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

    if not cfg['debug']:
        if multi_gpu:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
            
        losses_plot.append((np.mean(losses_train), iteration_index))
        plot_progress(losses_plot, save=True)
        
        torch.save(state_dict, f"model_state_last.pth")
