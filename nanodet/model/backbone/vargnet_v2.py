# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

__all__ = [
    "ConvModule2d",
    "ConvTransposeModule2d",
    "ConvUpsample2d",
    "SeparableConvModule2d",
    "SeparableGroupConvModule2d",
    "BasicVarGBlock",
    "BasicVarGBlockV2",
]


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
        act_layer: Optional[nn.Module] = None,
    ):
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
        act_layer: Optional[nn.Module] = None,
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
        act_layer: Optional[nn.Module] = None,
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
        dw_act_layer: Union[None, nn.Module] = None,
        pw_norm_layer: Union[None, nn.Module] = None,
        pw_act_layer: Union[None, nn.Module] = None,
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
        factor: float = 1,
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


class BasicVarGBlock(nn.Module):
    """
    A basic block for vargnetv2.

    Args:
        in_channels (int): Input channels.
        mid_channels (int): Mid channels.
        out_channels (int): Output channels.
        stride (int): Stride of basic block.
        kernel_size (int): Kernel size of basic block.
        padding (int): Padding of basic block.
        bias (bool): Whether to use bias in basic block.
        factor (int): Factor for channels expansion.
        group_base (int): Group base for group conv.
        merge_branch (bool): Whether to merge branch.
        dw_with_relu (bool): Whether to use relu in dw conv.
        pw_with_relu (bool): Whether to use relu in pw conv.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = True,
        factor: int = 2,
        group_base: int = 8,
        merge_branch: bool = False,
        dw_with_relu: bool = True,
        pw_with_relu: bool = True,
    ):
        super(BasicVarGBlock, self).__init__()
        self.merge_branch = merge_branch
        self.downsample = None
        if not (stride == 1 and in_channels == out_channels):
            self.downsample = SeparableGroupConvModule2d(
                in_channels,
                out_channels,
                kernel_size,
                bias=bias,
                padding=padding,
                factor=factor,
                groups=int(in_channels / group_base),
                stride=stride,
                dw_norm_layer=nn.BatchNorm2d(int(in_channels * factor)),
                dw_act_layer=nn.ReLU(inplace=True) if dw_with_relu else None,
                pw_norm_layer=nn.BatchNorm2d(out_channels),
            )

        if self.merge_branch:
            self.body_conv = SeparableGroupConvModule2d(
                in_channels,
                mid_channels,
                kernel_size,
                bias=bias,
                padding=padding,
                factor=factor,
                groups=int(in_channels / group_base),
                stride=stride,
                dw_norm_layer=nn.BatchNorm2d(int(in_channels * factor)),
                dw_act_layer=nn.ReLU(inplace=True) if dw_with_relu else None,
                pw_norm_layer=nn.BatchNorm2d(mid_channels),
            )
        self.conv = SeparableGroupConvModule2d(
            in_channels,
            mid_channels,
            kernel_size,
            bias=bias,
            padding=padding,
            factor=factor,
            groups=int(in_channels / group_base),
            stride=stride,
            dw_norm_layer=nn.BatchNorm2d(int(in_channels * factor)),
            dw_act_layer=nn.ReLU(inplace=True) if dw_with_relu else None,
            pw_norm_layer=nn.BatchNorm2d(mid_channels),
        )
        self.out_conv = SeparableGroupConvModule2d(
            mid_channels,
            out_channels,
            kernel_size,
            bias=bias,
            padding=padding,
            factor=factor,
            groups=int(mid_channels / group_base),
            stride=1,
            dw_norm_layer=nn.BatchNorm2d(int(mid_channels * factor)),
            dw_act_layer=nn.ReLU(inplace=True) if dw_with_relu else None,
            pw_norm_layer=nn.BatchNorm2d(out_channels),
        )

        self.merge_act = nn.ReLU(inplace=True)
        self.out_act = nn.ReLU(inplace=True) if pw_with_relu else None

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if self.merge_branch:
            body = self.body_conv(x)
            x = self.conv(x)
            x = x + body
            x = self.merge_act(x)
        else:
            x = self.conv(x)
            x = self.merge_act(x)

        out = self.out_conv(x)
        out = out + identity
        if self.out_act is not None:
            out = self.out_act(out)
        return out


class BasicVarGBlockV2(BasicVarGBlock):
    """
    A basic block for VargNASNet which inherits from BasicVarGBlock.

    The difference between BasicVarGBlockV2 and BasicVarGBlock is that
    `downsample` can be changed from `SeparableGroupConvModule2d` to
    `ConvModule2d`.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        stride (int): Stride of basic block.
        merge_branch (bool): Whether to merge branch.
        pw_with_relu (bool): Whether to use relu in pw conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        merge_branch: bool = False,
        pw_with_relu: bool = False,
        **kwargs,
    ):
        self.ch_change = False
        if in_channels != out_channels and stride == 1:
            self.ch_change = True
            merge_branch = False
        super(BasicVarGBlockV2, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            merge_branch=merge_branch,
            pw_with_relu=pw_with_relu,
            **kwargs,
        )
        if self.ch_change:
            self.downsample = ConvModule2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                norm_layer=nn.BatchNorm2d(out_channels),
            )


class OnePathResUnit(nn.Module):
    """
    ResUnit of One path.

    Args:
        dw_num_filter (int): Num filters of dw conv.
        group_base (int): Group base of group conv.
        pw_num_filter (int): Num filters of pw conv.
        pw_num_filter2 (int): Num filters of second pw conv.
        stride (int): Stride of group conv.
        is_dim_match (bool): Whether to use dim match.
        in_filter (:obj:'int', optional): Num filters of input. Default to
            None when in_filter == dw_num_filter
        use_bias (bool): Whether to use bias.
        pw_with_act (bool): Whether to use act of pw.
        factor (float): Factor of group conv.
    """

    def __init__(
        self,
        dw_num_filter,
        group_base,
        pw_num_filter,
        pw_num_filter2,
        stride,
        is_dim_match,
        in_filter=None,
        use_bias=False,
        pw_with_act=False,
        factor=2.0,
    ):
        super().__init__()
        assert dw_num_filter % group_base == 0
        assert pw_num_filter % group_base == 0

        if in_filter is None:
            in_filter = dw_num_filter
        if not is_dim_match:
            self.short_cut = SeparableGroupConvModule2d(
                in_channels=in_filter,
                dw_channels=dw_num_filter,
                out_channels=pw_num_filter2,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=use_bias,
                factor=factor,
                groups=dw_num_filter // group_base,
                dw_norm_layer=nn.BatchNorm2d(int(dw_num_filter * factor)),
                dw_act_layer=nn.ReLU(inplace=True),
                pw_norm_layer=nn.BatchNorm2d(pw_num_filter2),
            )

        self.sep1 = SeparableGroupConvModule2d(
            in_channels=in_filter,
            dw_channels=dw_num_filter,
            out_channels=pw_num_filter,
            kernel_size=3,
            stride=stride,
            padding=1,
            factor=factor,
            groups=dw_num_filter // group_base,
            bias=use_bias,
            dw_norm_layer=nn.BatchNorm2d(int(dw_num_filter * factor)),
            dw_act_layer=nn.ReLU(inplace=True),
            pw_norm_layer=nn.BatchNorm2d(pw_num_filter),
            pw_act_layer=nn.ReLU(inplace=True),
        )

        self.sep2 = SeparableGroupConvModule2d(
            in_channels=pw_num_filter,
            out_channels=pw_num_filter2,
            kernel_size=3,
            stride=1,
            padding=1,
            factor=factor,
            groups=pw_num_filter // group_base,
            bias=use_bias,
            dw_norm_layer=nn.BatchNorm2d(int(pw_num_filter * factor)),
            dw_act_layer=nn.ReLU(inplace=True),
            pw_act_layer=None,
            pw_norm_layer=nn.BatchNorm2d(pw_num_filter2),
        )
        self.sep2_relu = nn.ReLU(inplace=True) if pw_with_act else None

    def forward(self, x):
        short_cut = self.short_cut(x) if hasattr(self, "short_cut") else x

        out = self.sep1(x)
        out = self.sep2(out)

        out = out + short_cut
        if self.sep2_relu:
            out = self.sep2_relu(out)
        return out


class ExtendVarGNetFeatures(nn.Module):
    """
    Extend features.

    Args:
        prev_channel (int): Input channels.
        channels (list): Channels of output featuers.
        num_units (list): The number of units of each extend stride.
        group_base (int): The number of channels per group.
        factor (float, optional): Channel factor, by default 2.0
        dropout_kwargs (dict, optional): QuantiDropout kwargs,
            None means do not use drop, by default None
    """

    def __init__(
        self,
        prev_channel: int,
        channels: Union[List, int],
        num_units: Union[List, int],
        group_base: int,
        factor: float = 2.0,
        dropout_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        if dropout_kwargs is not None:
            self.dropout = nn.Dropout(**dropout_kwargs)

        channels = channels if isinstance(channels, Iterable) else [channels]
        num_units = num_units if isinstance(num_units, Iterable) else [num_units]

        for channel_i, num_unit_i in zip(channels, num_units):
            block = []
            assert num_unit_i >= 1
            for n_idx in range(num_unit_i):
                block.append(
                    OnePathResUnit(
                        dw_num_filter=prev_channel if n_idx == 0 else channel_i,
                        group_base=group_base,
                        pw_num_filter=channel_i,
                        pw_num_filter2=channel_i,
                        is_dim_match=n_idx != 0,
                        stride=2 if n_idx == 0 else 1,
                        pw_with_act=False,
                        use_bias=True,
                        factor=factor,
                    )
                )

            if dropout_kwargs is not None:
                block.append(nn.Dropout(**dropout_kwargs))
            self.blocks.append(nn.Sequential(*block))
            prev_channel = channel_i

    def forward(self, features):
        extend_features = list(features)
        if hasattr(self, "dropout"):
            extend_features[-1] = self.dropout(extend_features[-1])
        for block in self.blocks:
            extend_features.append(block(extend_features[-1]))

        return extend_features


class VargNetV2(nn.Module):
    """
    A module of vargnetv2.

    Args:
        alpha (float): Alpha for vargnetv2.
        group_base (int): Group base for vargnetv2.
        factor (int): Factor for channel expansion in basic block.
        out_stages (list): Indices for output.
        extend_features (bool): Whether to extend features.
        head_factor (int): Factor for channels expansion of stage1(mod2).
        model_type (str): Choose to use `VargNetV2` or `TinyVargNetV2`.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        group_base: int = 8,
        factor: int = 2,
        out_stages: Union[List, Tuple] = (2, 3, 4),
        extend_features: bool = False,
        head_factor: int = 1,
        model_type: str = "VargNetV2",
    ):
        super(VargNetV2, self).__init__()
        assert set(out_stages).issubset((0, 1, 2, 3, 4))
        self.out_stages = out_stages
        self.model_type = model_type.lower()
        assert self.model_type in ["vargnetv2", "tinyvargnetv2"], (
            f"`model_type` should be one of ['vargnetv2', 'tinyvargnetv2'],"
            f" but get {model_type}."
        )
        self.group_base = group_base
        self.factor = factor
        self.bias = True
        self.extend_features = extend_features
        self.head_factor = head_factor
        assert self.head_factor in [1, 2], "head_factor should be 1 or 2"
        # channel_list = [32, 32, 64, 128, 256]
        channel_list = [16, 32, 64, 128, 256]
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
            bias=self.bias,
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

    def forward(self, x):
        output = []
        for module in [self.mod1, self.mod2, self.mod3, self.mod4, self.mod5]:
            x = module(x)
            output.append(x)

        if self.extend_features:
            output = self.ext(output)
        output = [output[i] for i in self.out_stages]
        return output


if __name__ == "__main__":
    net = VargNetV2(
        alpha=2.0,
        group_base=8,
        factor=1,
        out_stages=(2, 3, 4),
        extend_features=False,
        head_factor=1,
        model_type="VargNetV2",
    )

    net.eval()
    inp = torch.randn(1, 3, 640, 640)
    out = net(inp)
    print([o.shape for o in out])

    # torch.onnx.export(
    #     net, inp, 'vargnet.onnx', opset_version=11
    # )
