# Copyright (c) Horizon Robotics. All rights reserved.

from typing import List, Tuple, Union

import torch.nn as nn

from .basic_module import ConvModule2d
from .basic_vargnet_module import VargDarkNetBlock


class VarGDarknet(nn.Module):
    def __init__(
        self,
        layers: List[int],
        filters: List[int],
        out_stages: Union[List, Tuple] = (),
    ):
        super(VarGDarknet, self).__init__()
        self.out_stages = out_stages
        self.conv1 = ConvModule2d(
            in_channels=3,
            out_channels=filters[0],
            kernel_size=3,
            padding=1,
            stride=1,
            norm_layer=nn.BatchNorm2d(filters[0]),
            act_layer=nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList()
        for i, layer in zip(range(len(layers)), layers):
            stage = nn.ModuleList()
            stage.append(
                ConvModule2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    norm_layer=nn.BatchNorm2d(filters[i + 1]),
                    act_layer=nn.ReLU(inplace=True),
                )
            )
            for _ in range(layer):
                stage.append(
                    VargDarkNetBlock(
                        in_channels=filters[i + 1],
                        out_channels=int(filters[i + 1] / 2),
                    )
                )
            stage = nn.Sequential(*stage)
            self.stages.append(stage)

    def forward(self, x):
        output = []
        x = self.conv1(x)
        output.append(x)
        for stage in self.stages:
            x = stage(x)
            output.append(x)
        if len(self.out_stages):
            return [output[idx] for idx in self.out_stages]
        return x


class VarGDarkNet53(nn.Module):
    def __init__(
        self,
        max_channels: int,
        stages: Union[List, Tuple] = (),
    ):
        super(VarGDarkNet53, self).__init__()
        if max_channels == 512:
            layers = [1, 2, 8, 8, 4]
            filters = [32, 64, 128, 256, 256, 512]
        else:
            layers = [1, 2, 8, 8, 4]
            filters = [32, 64, 128, 256, 512, 1024]

        self.model = VarGDarknet(layers=layers, filters=filters, out_stages=stages)

    def forward(self, x):
        return self.model(x)


class VarGDarkNetX(nn.Module):
    def __init__(
        self,
        ratio: float,
        out_stages: Union[List, Tuple] = (),
    ):
        super(VarGDarkNetX, self).__init__()
        layers = [1, 2, 8, 8, 4]
        # filters = [32, 64, 128, 256, 256, 512]
        filters = [32, 64, 128, 256, 512, 1024]
        filters = [int(f * ratio) for f in filters]

        self.model = VarGDarknet(layers=layers, filters=filters, out_stages=out_stages)

    def forward(self, x):
        return self.model(x)


__all__ = ["VarGDarknet", "VarGDarkNet53", "VarGDarkNetX"]
