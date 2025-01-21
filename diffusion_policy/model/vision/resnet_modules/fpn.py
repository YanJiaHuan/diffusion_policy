from typing import Callable, List, Optional

import torch.nn as nn
from torch import Tensor
from typing import Dict
from torchvision.ops import FrozenBatchNorm2d
from torchvision.ops.feature_pyramid_network import (
    ExtraFPNBlock,
    FeaturePyramidNetwork,
    LastLevelMaxPool
)
from torchvision.models._utils import IntermediateLayerGetter

from . import resnet


class BackboneWithFPN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone,
                                            return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.body(x)
        x = self.fpn(x)
        return x


def resnet_fpn_backbone(
    *,
    backbone_name: str,
    norm_layer: Callable[..., nn.Module] = FrozenBatchNorm2d,
    rgbd: bool = False,
    returned_layers: Optional[List[int]] = None,
) -> BackboneWithFPN:
    backbone = resnet.__dict__[backbone_name](norm_layer=norm_layer, rgbd=rgbd)
    return _resnet_fpn_extractor(backbone, returned_layers)


def _resnet_fpn_extractor(
    backbone: resnet.ResNet,
    returned_layers: Optional[List[int]] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
) -> BackboneWithFPN:

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(
        backbone, return_layers, in_channels_list, out_channels,
        norm_layer=norm_layer
    )