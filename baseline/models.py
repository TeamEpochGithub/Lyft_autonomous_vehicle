import torch
import torchvision

from torch import nn
from torchvision.models.resnet import resnet50
from typing import Dict

class BaselineModel(nn.Module):
    def __init__(self, conf: Dict):
        super().__init__()
        
        self.future_num_frames = conf["model_params"]["future_num_frames"]
        target_count = 2 * self.future_num_frames

        if conf["model_params"]["multi_mode"]:
            self.multi_mode = True
            target_count += 1   # One confidence per prediction
            target_count *= 3   # 3 predictions instead of 1
        else:
            self.multi_mode = False

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
        y = self.backbone(x)

        batches, _ = y.shape

        if self.multi_mode:
            pred, confidences = torch.split(x, self.future_num_frames * 3, dim=1)
            pred = pred.view(batches, 3, self.future_num_frames, 2)

            assert confidences.shape == (batches, 3)

            confidences = torch.softmax(confidences, dim=1)

            return pred, confidences
        else:
            return y

class MultiMode(nn.Module):
    def __init__():
        pass