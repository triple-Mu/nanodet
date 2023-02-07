# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .basic_module import ConvModule2d
from .basic_vargnet_module import BasicVarGBlock, ExtendVarGNetFeatures


class VargNetV2(nn.Module):
    """
    A module of vargnetv2.

    Args:
        model_type (str): Choose to use `VargNetV2` or `TinyVargNetV2`.
        alpha (float): Alpha for vargnetv2.
        group_base (int): Group base for vargnetv2.
        factor (float): Factor for channel expansion in basic block.
        bias (bool): Whether to use bias in module.
        extend_features (bool): Whether to extend features.
        stages (list): Return outputs by indices.
        head_factor (int): Factor for channels expansion of stage1(mod2).
    """

    def __init__(
        self,
        model_type: str = "VargNetV2",
        alpha: float = 1.0,
        group_base: int = 8,
        factor: float = 2,
        bias: bool = True,
        extend_features: bool = False,
        stages: Union[List, Tuple] = (),
        head_factor: int = 1,
    ):
        super(VargNetV2, self).__init__()
        self.model_type = model_type.lower()
        assert self.model_type in ["vargnetv2", "tinyvargnetv2"], (
            f"`model_type` should be one of ['vargnetv2', 'tinyvargnetv2'],"
            f" but get {model_type}."
        )
        self.group_base = group_base
        self.factor = factor
        self.bias = bias
        self.extend_features = extend_features
        self.stages = stages

        self.head_factor = head_factor
        assert self.head_factor in [1, 2], "head_factor should be 1 or 2"

        channel_list = [32, 32, 64, 128, 256]
        if self.model_type == "tinyvargnetv2":
            units = [1, 3, 4, 2]
        else:
            units = [1, 3, 7, 4]
        channel_list = [int(chls * alpha) for chls in channel_list]

        self.in_channels = channel_list[0]
        self.mod1 = ConvModule2d(
            3,
            channel_list[0],
            3,
            stride=2,
            padding=1,
            bias=bias,
            norm_layer=nn.BatchNorm2d(channel_list[0]),
            act_layer=nn.ReLU(inplace=True)
            if self.model_type == "tinyvargnetv2"
            else None,
        )

        head_factor = 2 if self.head_factor == 2 else 8 // group_base
        self.mod2 = self._make_stage(channel_list[1], units[0], head_factor, False)
        self.mod3 = self._make_stage(channel_list[2], units[1], self.factor)
        self.mod4 = self._make_stage(channel_list[3], units[2], self.factor)
        self.mod5 = self._make_stage(channel_list[4], units[3], self.factor)

        if extend_features:
            self.ext = ExtendVarGNetFeatures(
                prev_channel=channel_list[-1],
                channels=channel_list[-1],
                num_units=2,
                group_base=group_base,
            )

    def _make_stage(self, channels, repeats, factor, merge_branch=True):
        layers = []
        layers.append(
            BasicVarGBlock(
                self.in_channels,
                channels,
                channels,
                2,
                bias=self.bias,
                factor=factor,
                group_base=self.group_base,
                merge_branch=merge_branch,
            )
        )

        self.in_channels = channels
        for _ in range(1, repeats):
            layers.append(
                BasicVarGBlock(
                    channels,
                    channels,
                    channels,
                    1,
                    bias=self.bias,
                    factor=factor,
                    group_base=self.group_base,
                    merge_branch=False,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        output = []
        for module in [self.mod1, self.mod2, self.mod3, self.mod4, self.mod5]:
            x = module(x)
            output.append(x)

        if self.extend_features:
            output = self.ext(output)

        if len(self.stages):
            return [output[idx] for idx in self.stages]
        return x


def get_vargnetv2_stride2channels(
    alpha: float,
    channels: Optional[List[int]] = None,
    strides: Optional[List[int]] = None,
) -> Dict:
    """
    Get vargnet v2 stride to channel dict with giving channels and strides.

    Args:
        alpha: channel multipler.
        channels: base channel of each stride.
        strides: stride list corresponding to channels.

    Returns
        strides2channels: a stride to channel dict.
    """
    if channels is None:
        channels = [8, 8, 16, 32, 64, 64, 128, 256]
    if strides is None:
        strides = [2, 4, 8, 16, 32, 64, 128, 256]

    assert alpha in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    strides2channels = {}
    for s, c in zip(strides, channels):
        strides2channels[s] = int(alpha / 0.25) * c
    return strides2channels


class TinyVargNetV2(VargNetV2):
    """
    A module of TinyVargNetv2.

    Args:
        alpha (float): Alpha for tinyvargnetv2.
        group_base (int): Group base for tinyvargnetv2.
        factor (int): Factor for channel expansion in basic block.
        bias (bool): Whether to use bias in module.
        extend_features (bool): Whether to extend features.
        stages (list): Return outputs by indices.
        head_factor (int): Factor for channels expansion of stage1(mod2).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        group_base: int = 8,
        factor: int = 2,
        bias: bool = True,
        extend_features: bool = False,
        stages: Union[List, Tuple] = (),
        head_factor: int = 1,
    ):
        model_type = "TinyVargNetV2"

        super(TinyVargNetV2, self).__init__(
            model_type=model_type,
            alpha=alpha,
            group_base=group_base,
            factor=factor,
            bias=bias,
            extend_features=extend_features,
            stages=stages,
            head_factor=head_factor,
        )


class CocktailVargNetV2(VargNetV2):
    """CocktailVargNetV2.

    对 VargNetV2 进行了简单魔改.
    主要是去掉对 num_classes 作为 args 的要求和支持 top_layer 自定义.

    TODO(ziyang01.wang) 重构计划, 将相应的修改吸收到 VargNetV2 中.
    """

    def __init__(
        self,
        model_type: str = "VargNetV2",
        alpha: float = 1.0,
        group_base: int = 8,
        factor: int = 2,
        bias: bool = True,
        head_factor: int = 1,
        top_layer: Optional[nn.Module] = None,
    ):
        super(VargNetV2, self).__init__()
        self.model_type = model_type.lower()
        assert self.model_type in ["vargnetv2", "tinyvargnetv2"], (
            f"`model_type` should be one of ['vargnetv2', 'tinyvargnetv2'],"
            f" but get {model_type}."
        )
        self.group_base = group_base
        self.factor = factor
        self.bias = bias
        self.head_factor = head_factor
        assert self.head_factor in [1, 2], "head_factor should be 1 or 2"

        channel_list = [32, 32, 64, 128, 256]
        if self.model_type == "tinyvargnetv2":
            units = [1, 3, 4, 2]
        else:
            units = [1, 3, 7, 4]
        channel_list = [int(chls * alpha) for chls in channel_list]

        self.in_channels = channel_list[0]
        self.mod1 = ConvModule2d(
            3,
            channel_list[0],
            3,
            stride=2,
            padding=1,
            bias=bias,
            norm_layer=nn.BatchNorm2d(channel_list[0]),
            act_layer=nn.ReLU(inplace=True)
            if self.model_type == "tinyvargnetv2"
            else None,
        )

        head_factor = 2 if self.head_factor == 2 else 8 // group_base
        self.mod2 = self._make_stage(channel_list[1], units[0], head_factor, False)
        self.mod3 = self._make_stage(channel_list[2], units[1], self.factor)
        self.mod4 = self._make_stage(channel_list[3], units[2], self.factor)
        self.mod5 = self._make_stage(channel_list[4], units[3], self.factor)
        self.output = None if top_layer is None else top_layer

    def forward(self, x):
        output = []
        for module in [self.mod1, self.mod2, self.mod3, self.mod4, self.mod5]:
            x = module(x)
            output.append(x)
        if self.output is not None:
            x = self.output(x)
        return x


__all__ = ["VargNetV2", "get_vargnetv2_stride2channels", "TinyVargNetV2"]
