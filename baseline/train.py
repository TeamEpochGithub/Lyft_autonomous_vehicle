import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
# from prettytable import PrettyTable
# from pathlib import Path

import os
import argparse
from tqdm import tqdm

import models
from models import BaselineModel

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, help="Where to find the competition data")
    parser.add_argument("--config", type=str, help="Location of the config file")

    args = parser.parse_args()

    # Load config
    os.environ["L5KIT_DATA_FOLDER"] = args.input_dir
    dm = LocalDataManager(None)
    cfg = load_config_data(args.config)

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
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
  
    if cfg["train_params"]["loss"] == "mse":
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
        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)

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