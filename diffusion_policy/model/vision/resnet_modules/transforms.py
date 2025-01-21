import torch
import torch.nn as nn
from torchvision.transforms import Normalize, Resize


class SAM2Transforms(nn.Module):
    def __init__(self, resolution):
        """
        Transforms for SAM2.
        """
        super().__init__()
        self.resolution = resolution
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transforms = torch.jit.script(
            nn.Sequential(
                Resize((self.resolution, self.resolution)),
                Normalize(self.mean, self.std),
            )
        )

    def __call__(self, x):
        return self.transforms(x)

    def forward_batch(self, img_list):
        img_batch = self.transforms(img_list)
        return img_batch
