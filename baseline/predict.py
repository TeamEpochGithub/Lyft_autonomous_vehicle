import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50
from torch.cuda.amp import autocast

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
from contextlib import nullcontext

import argparse
from tqdm import tqdm

import models
from models import BaselineModel

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, help="Where to find the competition data")
    parser.add_argument("--config", type=str, help="Location of the config file")
    parser.add_argument("--weight-file", type=str, help="path tho the file containing the weights for the model")

    args = parser.parse_args()

    weight_file = args.weight_file

    # Load config
    os.environ["L5KIT_DATA_FOLDER"] = args.input_dir
    dm = LocalDataManager(None)
    cfg = load_config_data(args.config)

    multi_mode = cfg["model_params"]["multi_mode"]
    
    # Generate and load chopped dataset
    rasterizer = build_rasterizer(cfg, dm)

    num_frames_to_chop = 100
    eval_cfg = cfg["val_data_loader"]
    
    eval_zarr_path = dm.require(eval_cfg["key"])
    eval_mask_path = dm.require("scenes/mask.npz")

    eval_zarr = ChunkedDataset(eval_zarr_path).open()
    eval_mask = np.load(eval_mask_path)["arr_0"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Init dataset and load mask
    eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer, agents_mask=eval_mask)
    eval_dataloader = DataLoader(eval_dataset, shuffle=eval_cfg["shuffle"], batch_size=eval_cfg["batch_size"], 
                                num_workers=eval_cfg["num_workers"])
    print(eval_dataset)

    # Create model
    model = BaselineModel(cfg)
    model.to(device)
    model.load_state_dict(
        torch.load(weight_file, map_location=device)
    )

    # Eval loop
    with torch.no_grad():
        model.eval()

        # store information for evaluation
        future_coords_offsets_pd = []
        timestamps = []
        agent_ids = []
        all_confidences = []

        print("Start eval loop")
        
        progress_bar = tqdm(eval_dataloader)

        j = 0

        for data in progress_bar:
            if False:
                if j > 10:
                    break
                else:
                    j += 1

            target_positions = data["target_positions"].to(device)

            with autocast() if torch.cuda.is_available() else nullcontext:
                if multi_mode:
                    predictions, confidences = model(data["image"].to(device))
                    predictions = predictions.reshape(target_positions.shape + (3,))

                    confidences = confidences.cpu().numpy()
                else:
                    predictions = model(data["image"].to(device)).reshape(target_positions.shape)
                
            agents_coords = predictions.cpu().numpy()
            # print(agents_coords.shape)

            # convert agent coordinates into world offsets
            world_from_agents = data["world_from_agent"].numpy()
            centroids = data["centroid"].numpy()
            coords_offset = []
            
            for agent_coords, world_from_agent, centroid in zip(agents_coords, world_from_agents, centroids):
                if multi_mode:
                    predictions = np.zeros((3, 50, 2))
                    for i in range(3):
                        predictions[i] = (transform_points(agent_coords[:, :, i], world_from_agent) - centroid[:2])

                    coords_offset.append(predictions)
                    # print("coords_offset[-1].shape =", coords_offset[-1].shape)
                    # print("predictions[-1].shape =", predictions[-1].shape)
                else:
                    coords_offset.append(transform_points(agent_coords, world_from_agent) - centroid[:2])

            future_coords_offsets_pd.append(np.stack(coords_offset))
            timestamps.append(data["timestamp"].numpy().copy())
            agent_ids.append(data["track_id"].numpy().copy())
            if multi_mode:
                all_confidences.append(confidences)

        print("Done eval loop")

    pred_path = f"submission.csv"

    write_pred_csv(pred_path,
                timestamps=np.concatenate(timestamps),
                track_ids=np.concatenate(agent_ids),
                coords=np.concatenate(future_coords_offsets_pd),
                confs=np.concatenate(all_confidences) if multi_mode else None
                )

    print("Written submission csv. Everything is done.")
