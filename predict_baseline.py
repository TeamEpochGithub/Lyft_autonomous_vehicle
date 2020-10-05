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
from l5kit.evaluation import write_pred_csv
from l5kit.rasterization import build_rasterizer
from itertools import takewhile
import argparse
from baseline_model import LyftModel

default_cfg = {
    'format_version': 4,
    'model_params': {
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1
    },
    
    'raster_params': {
        'raster_size': [300, 300],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },
    
    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 16,
        'shuffle': False,
        'num_workers': 8
    }
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, help="where to find the competition data")
    parser.add_argument("--weight-file", type=str, help="path tho the file containing the weights for the model")
    args = parser.parse_args()

    DIR_INPUT = args.input_dir
    WEIGHT_FILE = args.weight_file
    os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT

    dm = LocalDataManager(None)
    cfg = default_cfg.copy()

    # ===== INIT DATASET
    test_cfg = cfg["test_data_loader"]

    # Rasterizer
    rasterizer = build_rasterizer(cfg, dm)

    # Test dataset/dataloader
    test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()
    test_mask = np.load(f"{DIR_INPUT}/scenes/mask.npz")["arr_0"]
    test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
    test_dataloader = DataLoader(test_dataset,
                                shuffle=test_cfg["shuffle"],
                                batch_size=test_cfg["batch_size"],
                                num_workers=test_cfg["num_workers"])


    print(test_dataloader)

    # ==== INIT MODEL
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LyftModel(cfg)
    model.to(device)

    if WEIGHT_FILE is not None:
        # Saved state dict from the training notebook
        model_state = torch.load(WEIGHT_FILE, map_location=device)
        model.load_state_dict(model_state)

        print(device)

    model.eval()

    future_coords_offsets_pd = []
    timestamps = []
    agent_ids = []

    with torch.no_grad():
        dataiter = tqdm(test_dataloader)
        
        for data in dataiter:
            inputs = data["image"].to(device)
            target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
            targets = data["target_positions"].to(device)

            outputs = model(inputs).reshape(targets.shape)
            
            future_coords_offsets_pd.append(outputs.cpu().numpy().copy())
            timestamps.append(data["timestamp"].numpy().copy())
            agent_ids.append(data["track_id"].numpy().copy())

    write_pred_csv('submission.csv',
        timestamps=np.concatenate(timestamps),
        track_ids=np.concatenate(agent_ids),
        coords=np.concatenate(future_coords_offsets_pd))
