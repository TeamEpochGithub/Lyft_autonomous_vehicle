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

import models
from models import BaselineModel

import loss_functions

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

    if args.multi_gpu:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    if multi_mode:
        criterion = loss_functions.pytorch_neg_multi_log_likelihood_batch
    elif cfg["train_params"]["loss"] == "mse":
        criterion = nn.MSELoss(reduction="none")

    # Train loop
    print("Start train loop")
    
    tr_it = iter(train_dataloader)
    progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
    losses_train = []
    model.train()
    torch.set_grad_enabled(True)
    
    for itr in progress_bar:
        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dataloader)
            data = next(tr_it)

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

        if (itr+1) % cfg['train_params']['checkpoint_every_n_steps'] == 0 and not cfg['debug']:
            torch.save(model.state_dict(), f'model_state_{itr}.pth')
        
        losses_train.append(loss.item())
        losses_train = losses_train[-100:]
        progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

    if not cfg['debug']:
        torch.save(model.state_dict(), f"model_state_last.pth")