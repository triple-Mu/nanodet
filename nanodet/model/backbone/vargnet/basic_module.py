# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class ConvModule2d(nn.Sequential):
    """
    A conv block that bundles conv/norm/activation layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool): Same as nn.Conv2d.
        padding_mode (str): Same as nn.Conv2d.
        norm_layer (nn.Module): Normalization layer.
        act_layer (nn.Module): Activation layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = nn.ReLU(inplace=True),
    ):
        if isinstance(norm_layer, nn.BatchNorm2d):
            bias = False
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        conv_list = [conv, norm_layer, act_layer]
        self.conv_list = [layer for layer in conv_list if layer is not None]
        super(ConvModule2d, self).__init__(*self.conv_list)

    def forward(self, x):
        out = super().forward(x)
        return out


class ConvTransposeModule2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: bool = 1,
        padding_mode: str = "zeros",
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = nn.ReLU(inplace=True),
    ):
        """Transposed convolution, followed by normalization and activation.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (Union[int, Tuple[int, int]]): kernel size.
            stride (Union[int, Tuple[int, int]], optional): conv stride.
                Defaults to 1.
            padding (Union[int, Tuple[int, int]], optional): conv padding.
                dilation * (kernel_size - 1) - padding zero-padding will be
                added to the input. Defaults to 0.
            output_padding (Union[int, Tuple[int, int]], optional):
                additional size added to the output. Defaults to 0.
            groups (int, optional): number of blocked connections from input
                to output. Defaults to 1.
            bias (bool, optional): whether to add learnable bias.
                Defaults to True.
            dilation (bool, optional): kernel dilation. Defaults to 1.
            padding_mode (str, optional): same as conv2d. Defaults to 'zeros'.
            norm_layer (Optional[nn.Module], optional): normalization layer.
                Defaults to None.
            act_layer (Optional[nn.Module], optional): activation layer.
                Defaults to None.
        """
        super().__init__()
        if isinstance(norm_layer, nn.BatchNorm2d):
            bias = False
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
        )
        self.norm = norm_layer
        self.act = act_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvUpsample2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = nn.ReLU(inplace=True),
    ):
        """Conv upsample module.

        Different from ConvTransposeModule2d, this module does conv2d,
        followed by an upsample layer. The final effect is almost the same,
        but one should pay attention to the output size.

        Args:
            in_channels (int): same as nn.Conv2d.
            out_channels (int): same as nn.Conv2d.
            kernel_size (Union[int, Tuple[int, int]]): same as nn.Conv2d.
            stride (Union[int, Tuple[int, int]], optional): Upsample stride.
                Defaults to 1.
            padding (Union[int, Tuple[int, int]], optional): same as nn.Conv2d.
                Defaults to 0.
            dilation (Union[int, Tuple[int, int]], optional): same as
                nn.Conv2d. Defaults to 1.
            groups (int, optional): same as nn.Conv2d. Defaults to 1.
            bias (bool, optional): same as nn.Conv2d. Defaults to True.
            padding_mode (str, optional): same as nn.Conv2d.
                Defaults to "zeros".
            norm_layer (Optional[nn.Module], optional): normalization layer.
                Defaults to None.
            act_layer (Optional[nn.Module], optional): activation layer.
                Defaults to None.
        """
        super().__init__()
        if isinstance(norm_layer, nn.BatchNorm2d):
            bias = False
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            1,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.norm = norm_layer
        self.act = act_layer
        self.up = nn.Upsample(scale_factor=stride, recompute_scale_factor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        x = self.up(x)
        return x


class SeparableConvModule2d(nn.Sequential):
    """
    Depthwise sparable convolution module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        bias (bool): Same as nn.Conv2d.
        padding_mode (str): Same as nn.Conv2d.
        dw_norm_layer (nn.Module): Normalization layer in dw conv.
        dw_act_layer (nn.Module): Activation layer in dw conv.
        pw_norm_layer (nn.Module): Normalization layer in pw conv.
        pw_act_layer (nn.Module): Activation layer in pw conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        dw_norm_layer: Union[None, nn.Module] = None,
        dw_act_layer: Union[None, nn.Module] = nn.ReLU(inplace=True),
        pw_norm_layer: Union[None, nn.Module] = None,
        pw_act_layer: Union[None, nn.Module] = nn.ReLU(inplace=True),
    ):
        super(SeparableConvModule2d, self).__init__(
            ConvModule2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                in_channels,
                bias,
                padding_mode,
                dw_norm_layer,
                dw_act_layer,
            ),
            ConvModule2d(
                in_channels,
                out_channels,
                1,
                bias=bias,
                norm_layer=pw_norm_layer,
                act_layer=pw_act_layer,
            ),
        )


class SeparableGroupConvModule2d(nn.Sequential):
    """
    Separable group convolution module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        dw_channels (:obj:'int', optional): Number of dw conv output channels.
            Default to None when dw_channels == in_channels.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool): Same as nn.Conv2d.
        padding_mode (str): Same as nn.Conv2d.
        dw_norm_layer (nn.Module): Normalization layer in group conv.
        dw_act_layer (nn.Module): Activation layer in group conv.
        pw_norm_layer (nn.Module): Normalization layer in pw conv.
        pw_act_layer (nn.Module): Activation layer in pw conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        dw_channels: Optional[int] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        factor: float = 1.0,
        bias: bool = True,
        padding_mode: str = "zeros",
        dw_norm_layer: Union[None, nn.Module] = None,
        dw_act_layer: Union[None, nn.Module] = None,
        pw_norm_layer: Union[None, nn.Module] = None,
        pw_act_layer: Union[None, nn.Module] = None,
    ):
        if dw_channels is None:
            dw_channels = in_channels

        super(SeparableGroupConvModule2d, self).__init__(
            ConvModule2d(
                in_channels,
                int(dw_channels * factor),
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                padding_mode,
                dw_norm_layer,
                dw_act_layer,
            ),
            ConvModule2d(
                int(dw_channels * factor),
                out_channels,
                1,
                bias=bias,
                norm_layer=pw_norm_layer,
                act_layer=pw_act_layer,
            ),
        )


class VargDarkNetBlock(nn.Module):
    """
    A basic block for vargdarknet.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super(VargDarkNetBlock, self).__init__()
        assert in_channels == out_channels * 2, f"{in_channels} != 2 * {out_channels}"
        self.conv1 = ConvModule2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=nn.BatchNorm2d(out_channels),
            act_layer=nn.ReLU(inplace=True),
        )
        self.conv2 = SeparableGroupConvModule2d(
            in_channels=out_channels,
            out_channels=out_channels * 2,
            kernel_size=5,
            stride=1,
            padding=2,
            factor=1,
            groups=8,
            dw_norm_layer=nn.BatchNorm2d(out_channels),
            dw_act_layer=None,
            pw_norm_layer=nn.BatchNorm2d(out_channels * 2),
            pw_act_layer=None,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x) + residual
        x = self.relu(x)
        return x


__all__ = [
    "ConvModule2d",
    "ConvTransposeModule2d",
    "ConvUpsample2d",
    "SeparableConvModule2d",
    "SeparableGroupConvModule2d",
    "VargDarkNetBlock",
]
