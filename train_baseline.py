import numpy as np
import os
import torch

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18
from tqdm import tqdm
from typing import Dict

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from baseline_model import LyftModel

default_cfg = {
    'debug': False,
    'format_version': 4,
    'model_params': {
        'model_architecture': 'resnet18',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1
    },
    
    'raster_params': {
        'raster_size': [224, 224],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic', # maybe semantic + box is better?
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },
    
    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 8
    },
    
    'train_params': {
        'checkpoint_every_n_steps': 5000,
    }
}

if __name__ == "__main__":
    cfg = default_cfg.copy()

    # set env variable for data
    # os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
    dm = LocalDataManager(None)

    print("DEBUG =", cfg['debug'])

    if not 'max_num_steps' in cfg['train_params']:
        cfg['train_params']['max_num_steps'] = 100 if cfg['debug'] else 50000
    
    print("cfg =")
    print(cfg)


    # ===== INIT DATASET
    train_cfg = cfg["train_data_loader"]

    # Rasterizer
    rasterizer = build_rasterizer(cfg, dm)

    # Train dataset/dataloader
    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
    train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    train_dataloader = DataLoader(train_dataset,
                                shuffle=train_cfg["shuffle"],
                                batch_size=train_cfg["batch_size"],
                                num_workers=train_cfg["num_workers"])

    print(train_dataset)

    # ==== INIT MODEL
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = LyftModel(cfg)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Later we have to filter the invalid steps.
    criterion = nn.MSELoss(reduction="none")

    print("device =", device)


    # ==== TRAIN LOOP
    tr_it = iter(train_dataloader)

    if False:
        print('len(tr_it) =',len(tr_it))

        raise Exception()

    progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
    losses_train = []

    for itr in progress_bar:

        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dataloader)
            data = next(tr_it)

        model.train()
        torch.set_grad_enabled(True)

        # Forward pass
        inputs = data["image"].to(device)
        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
        targets = data["target_positions"].to(device)

        outputs = model(inputs).reshape(targets.shape)
        loss = criterion(outputs, targets)

        # not all the output steps are valid, but we can filter them out from the loss using availabilities
        loss = loss * target_availabilities
        loss = loss.mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_train.append(loss.item())

        if (itr+1) % cfg['train_params']['checkpoint_every_n_steps'] == 0 and not cfg['debug']:
            torch.save(model.state_dict(), f'model_state_{itr}.pth')

        progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train[-100:])}")

    if not cfg['debug']:
        torch.save(model.state_dict(), f'model_state_last.pth')
    cfg['debug']
