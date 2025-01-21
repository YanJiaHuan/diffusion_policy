import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from .resnet_modules import resnet_fpn_backbone as fpn


class ResNet_FPN_model(torch.nn.Module):
    def __init__(self, name, pretrained=False):
        super(ResNet_FPN_model, self).__init__()
        self.model = resnet_fpn_backbone(name, pretrained=pretrained)

    def forward(self, x):
        features = self.model(x)
        embeddings = []

        for level, feature in features.items():
            embedding = F.adaptive_avg_pool2d(
                feature, (1, 1)).reshape(feature.size(0), -1)
            embeddings.append(embedding)

        concatenated_embeddings = torch.cat(embeddings, dim=1)

        return concatenated_embeddings


class ResNetFPN(nn.Module):
    def __init__(self, name, rgbd=False):
        super(ResNetFPN, self).__init__()
        self.model = fpn(backbone_name=name, rgbd=rgbd)

    def forward(self, x):
        features = self.model(x)
        embeddings = []

        for level, feature in features.items():
            embedding = F.adaptive_avg_pool2d(
                feature, (1, 1)).reshape(feature.size(0), -1)
            embeddings.append(embedding)

        concatenated_embeddings = torch.cat(embeddings, dim=1)

        return concatenated_embeddings