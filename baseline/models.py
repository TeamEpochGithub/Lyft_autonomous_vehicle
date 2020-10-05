import torch
import torchvision

from torch import nn
from torchvision.models.resnet import resnet50
from typing import Dict

class BaselineModel():
    def __init__(self, conf: Dict):
        target_count = 2 * conf["model_params"]["future_num_frames"]
        history_channel_count = (conf["model_params"]["history_num_frames"] + 1) * 2
        total_channel_count = 3 + history_channel_count

        if conf["model_params"]["model_architecture"] == "resnet50":
            backbone = resnet50(pretrained=conf["model_params"]["pretrained"])

            backbone.conv1 = nn.Conv2d(
                total_channel_count,
                backbone.conv1.out_channels,
                kernel_size=backbone.conv1.kernel_size,
                stride=backbone.conv1.stride,
                padding=backbone.conv1.padding,
                bias=conf["model_params"]["first_layer_bias"], # Maybe True is better?
            )

            backbone.fc = nn.Linear(in_features=2048, out_features=target_count)

            self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)