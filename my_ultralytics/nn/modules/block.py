# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock
from my_ultralytics.utils.torch_utils import fuse_conv_and_bn
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import cv2
from torch import Tensor
import math
from my_ultralytics.utils import LOGGER

def debug_tensor(tensor_name: str, tensor: torch.Tensor):
    """
    A robust debugging function that uses the Ultralytics LOGGER.
    This will ensure the output is displayed correctly in the training environment.
    """
    if tensor is None:
        LOGGER.info(f"--- [DEBUG] {tensor_name}: Tensor is None ---")
        return
    # 准备日志信息
    log_message = [f"--- [DEBUG] Inspecting: {tensor_name} ---"]
    log_message.append(f"    - Shape: {tensor.shape}")
    log_message.append(f"    - Dtype: {tensor.dtype}")
    
    # 检查 NaN/Inf
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    if has_nan or has_inf:
        log_message.append(f"    - !!! CRITICAL: Anomaly DETECTED !!!")
        log_message.append(f"    - Has NaN: {has_nan.item()}")
        log_message.append(f"    - Has Inf: {has_inf.item()}")
    else:
        # 如果没有异常，我们可以打印一些统计数据
        # 为了避免在GPU上计算统计数据引入额外问题，我们先移动到CPU
        stats_tensor = tensor.detach().to(torch.float32).cpu()
        log_message.append(f"    - Stats (on CPU, as float32):")
        log_message.append(f"        - Max:  {stats_tensor.max().item():.6f}")
        log_message.append(f"        - Min:  {stats_tensor.min().item():.6f}")
        log_message.append(f"        - Mean: {stats_tensor.mean().item():.6f}")
        log_message.append(f"        - Std:  {stats_tensor.std().item():.6f}")
        
    log_message.append("--------------------")
    # 使用LOGGER.info()来打印，确保它能被框架捕获
    LOGGER.info("\n" + "\n".join(log_message))

def inspect_forward_output(module, input_tensor, output_tensor):
    """
    一个前向传播Hook函数，用于打印和检查模块输出的详细信息。
    """
    try:
        print(f"\n{'='*40}")
        print(f"--- [HOOK] Inspecting Forward Output of: {module.__class__.__name__} ---")
        # 1. 核心检查：是否存在NaN或Inf
        has_nan = torch.isnan(output_tensor).any()
        has_inf = torch.isinf(output_tensor).any()
        if has_nan or has_inf:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! CRITICAL: NaN or Inf DETECTED in the forward pass output!")
            print(f"!!! Has NaN: {has_nan.item()}")
            print(f"!!! Has Inf: {has_inf.item()}")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        else:
            print("✅ Output tensor is clean (no NaN/Inf).")
        # 2. 打印输出张量的基本信息
        print(f"    - Output Tensor Shape: {output_tensor.shape}")
        print(f"    - Output Tensor Dtype: {output_tensor.dtype}")
        # 3. 打印关键的统计数据
        if output_tensor.numel() > 0:
            # 使用 .detach() 来避免Hook影响梯度计算
            max_val = torch.max(output_tensor.detach())
            min_val = torch.min(output_tensor.detach())
            mean_val = torch.mean(output_tensor.detach())
            std_val = torch.std(output_tensor.detach())
            
            print(f"    - Statistics:")
            print(f"        - Max Value:  {max_val.item():.6f}")
            print(f"        - Min Value:  {min_val.item():.6f}")
            print(f"        - Mean Value: {mean_val.item():.6f}")
            print(f"        - Std Dev:    {std_val.item():.6f}")
            if std_val.item() < 1e-8 and output_tensor.numel() > 1:
                print("\n    ⚠️ WARNING: Standard deviation is near-zero!")
                print("        This can cause issues with downstream BatchNorm layers.\n")
        else:
            print("    - Output tensor is empty.")
        print(f"--- [HOOK] Inspection Finished ---")
        print(f"{'='*40}\n")
    except Exception as e:
        print(f"!!! Error during Hook execution: {e}")
        
__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
    "S_UniRepLKNetBlock",
    "L_UniRepLKNetBlock",
    "Smak_Block",
    "Lark_Block",
    "RepViTBlock",
    "DenseRepViTBlock",
    "DenseRepViTBlock_",
    "C3k",
    "C3k2",
    "C2PSA",
    "DeformableViTBlock",
    "Eage_detect",
    "Edge_Emphasize",
    "DenseRepViTBlock_EGA",
    "RepViTBlock_ECA",
    "RepViTBlock_ECA_Att",
    "EG_stem",
    "EGA",
    "EGA_att",
    "ConcatShuffleConv",
    "Edge_guide",
    "Fuse_Features",
    "CSP_DenseRepViTBlock",
    "CSP_DenseRepViTBlock_"
)



def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
               attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)
def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)

# class Conv2d_BN(torch.nn.Sequential):
#     def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
#                  groups=1, bn_weight_init=1, resolution=-10000):
#         super().__init__()
#         self.add_module('c', torch.nn.Conv2d(
#             a, b, ks, stride, pad, dilation, groups, bias=False))
#         self.add_module('bn', torch.nn.BatchNorm2d(b))
#         torch.nn.init.constant_(self.bn.weight, bn_weight_init)
#         torch.nn.init.constant_(self.bn.bias, 0)
    
# def fuse_bn(conv, bn):
#     conv_bias = 0 if conv.bias is None else conv.bias
#     std = (bn.running_var + bn.eps).sqrt()
#     return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std

# def convert_dilated_to_nondilated(kernel, dilate_rate):
#     identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
#     if kernel.size(1) == 1:
#         #   This is a DW kernel
#         dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
#         return dilated
#     else:
#         #   This is a dense or group-wise (but not DW) kernel
#         slices = []
#         for i in range(kernel.size(1)):
#             dilated = F.conv_transpose2d(kernel[:,i:i+1,:,:], identity_kernel, stride=dilate_rate)
#             slices.append(dilated)
#         return torch.cat(slices, dim=1)

# def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
#     large_k = large_kernel.size(2)
#     dilated_k = dilated_kernel.size(2)
#     equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
#     equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
#     rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
#     merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
#     return merged_kernel

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            cv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(cv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Contrastive Head for YOLO-World compute the region-text scores according to the similarity between image and text
    features.
    """

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        self.bias = nn.Parameter(torch.zeros([]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(nn.Module):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcut option, groups and expansion
        ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass through RepBottleneck layer."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepCSP(nn.Module):
    """Rep CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through RepCSP layer."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class Silence(nn.Module):
    """Silence."""

    def __init__(self):
        """Initializes the Silence module."""
        super(Silence, self).__init__()

    def forward(self, x):
        """Forward pass through Silence layer."""
        return x


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.conv(x) + self.conv1(x))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [2,2,2,2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1

class RepVGGDW_ViT(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)
    
    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)
    
    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [1,1,1,1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv

class CIB(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            Conv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else RepVGGDW(2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv1(x) if self.add else self.cv1(x)

class C2fCIB(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8,
                 attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (
            (q.transpose(-2, -1) @ k) * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()

        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = F.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)

class PSA(nn.Module):

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert(c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        
        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(
            Conv(self.c, self.c*2, 1),
            Conv(self.c*2, self.c, 1, act=False)
        )
        
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))

class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        return self.cv2(self.cv1(x))

class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        '''
        根据卷积核大小（kernel_size），为膨胀卷积选择不同的卷积核尺寸和膨胀率（dilates）。
        膨胀卷积通过在卷积核元素之间插入间隙来扩大感受野，具体感受野大小由膨胀率决定。
        例如，当 kernel_size 为 17 时，使用 5、9 和 3 大小的卷积核，并以不同的膨胀率应用卷积
        '''
        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')
        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))
    '''
    前向传播：
    如果当前处于推理模式（即没有 origin_bn 属性），则只使用标准卷积 lk_origin 进行卷积操作，直接返回卷积结果。
    如果处于训练模式（origin_bn 存在），首先对输入特征图 x 应用标准卷积和 BN 层。
    然后，遍历膨胀卷积核尺寸和膨胀率，分别对输入特征图 x 应用对应的卷积和 BN 操作，并将其结果与前面的结果相加，从而融合膨胀卷积和标准卷积的特征。
    '''
    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out
    '''
    合并膨胀卷积和标准卷积 merge_dilated_branches：
    首先，将标准卷积 lk_origin 和其对应的 BN 层融合为一个等效的卷积层，使用 fuse_bn 函数将卷积权重和BN权重结合。
    然后，遍历所有膨胀卷积核和其对应的 BN 层，逐个将它们合并到标准卷积的权重中。merge_dilated_into_large_kernel 函数会将膨胀卷积核转换为非膨胀卷积，并将其与大核卷积融合。
    最终，将所有卷积核合并为一个单一的卷积层 merged_conv，并删除所有多余的卷积和 BN 分支，从而简化推理阶段的计算。
    '''
    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            '''
            这一段代码的作用是通过 __delattr__ 方法删除在初始化时为膨胀卷积和批归一化（Batch Normalization，BN）创建的属性。
            它位于 merge_dilated_branches 方法的最后部分，目的是在重参数化完成后，清理掉不再需要的卷积层和BN层，从而简化模型结构，减少内存占用
            self.kernel_sizes 和 self.dilates: 这些是类中定义的膨胀卷积核尺寸和相应的膨胀率，分别存储了不同尺寸的卷积核和膨胀卷积的膨胀因子。

            zip(self.kernel_sizes, self.dilates): 通过 zip 函数将卷积核的大小和相应的膨胀率一一配对。例如，self.kernel_sizes = [5, 9, 3] 和 self.dilates = [1, 2, 4]，则它们会形成如下配对：(5, 1)、(9, 2)、(3, 4)。

            self.__delattr__: 这是 Python 提供的用于删除对象属性的函数。通过 __delattr__，可以动态删除类实例中的指定属性
            dil_conv_k{}_{}'.format(k, r): 这个格式化字符串对应膨胀卷积层的名称，例如，如果 k = 5 且 r = 1，那么 dil_conv_k5_1 对应的是之前为卷积核大小为5、膨胀率为1的膨胀卷积层。

            dil_bn_k{}_{}'.format(k, r): 类似地，这个字符串对应膨胀卷积后接的批归一化层。例如，dil_bn_k5_1 对应的是卷积核为5、膨胀率为1的膨胀卷积后的BN层。
            '''
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))

class NCHWtoNHWC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


class NHWCtoNCHW(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)

class GRNwithNHWC(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, H, W, C)
    """
    def __init__(self, dim, use_bias=True):
        super() .__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x

class L_UniRepLKNetBlock(nn.Module):
    def __init__(self,
                 c1,
                 c2,
                 num=0,
                 judge=True,
                 e=0.5,
                 s=1,
                 k=17,
                 shortcut=True,
                 deploy=False,
                 attempt_use_lk_impl=True,
                 use_sync_bn=False,
                 layer_scale_init_value=1e-5
                 ):
        super().__init__()
        assert c1 == c2
        self.c_ = int(c1*e)
        self.c__ = int((1-e)*c1)
        dilated_block = DilatedReparamBlock(self.c_,17,deploy=deploy,attempt_use_lk_impl=attempt_use_lk_impl)
        self.cv = dilated_block
        self.pwconv1 = nn.Sequential(
            NCHWtoNHWC(),
            nn.Linear(self.c_,self.c_*2)
        )
        self.act = nn.Sequential(
            nn.GELU(),
            GRNwithNHWC(self.c_*2, use_bias=not deploy)
        )
        if deploy:
            self.pwconv2 = nn.Sequential(
                nn.Linear(self.c_*2, self.c_),
                NHWCtoNCHW()
            )
        else:
            self.pwconv2 = nn.Sequential(
                nn.Linear(self.c_*2, self.c_, bias=False),
                NHWCtoNCHW(),
                get_bn(self.c_, use_sync_bn=use_sync_bn)
            )
        if deploy or k == 0:
            self.norm = nn.Identity()
        else:
            self.norm = get_bn(self.c_)
        self.se = SEBlock(self.c_, self.c_ // 2)
        self.ffn = nn.Sequential(
            self.pwconv1,
            self.act,
            self.pwconv2
)

        self.shortcut = nn.Sequential(Conv(c1, c2, k=1, s=s, act=False)) if s != 1 or c1 != c2 else nn.Identity()
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(self.c_),
            requires_grad=True
        ) if not deploy else None



    def forward(self,x):
        y1,y2 = x.split((self.c_,self.c__),dim=1)
        # print("Input shape:", x.shape)
        y = self.se(self.norm(self.cv(y1)))
        # print("Input after:", x.shape)
        
        y = self.ffn(y)
        # print("Input shape afterffn:", x.shape)

        if self.gamma is not None:
            y = self.gamma.view(1, -1, 1, 1) * y
        
        y1 = y
        y = torch.cat((y1,y2),1)
        return y + self.shortcut(x)

    def reparameterize(self):
        if hasattr(self.cv, 'merge_dilated_branches'):
            self.cv.merge_dilated_branches()

        if hasattr(self.norm, 'running_var'):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            if hasattr(self.cv, 'lk_origin'):
                self.cv.lk_origin.weight.data *= (self.norm.weight / std).view(-1, 1, 1, 1)
                self.cv.lk_origin.bias.data = self.norm.bias + (
                        self.cv.lk_origin.bias - self.norm.running_mean) * self.norm.weight / std
            else:
                conv = nn.Conv2d(self.cv.in_channels, self.cv.out_channels, self.cv.kernel_size,
                             padding=self.cv.padding, groups=self.cv.groups, bias=True)
                conv.weight.data = self.cv.weight * (self.norm.weight / std).view(-1, 1, 1, 1)
                conv.bias.data = self.norm.bias - self.norm.running_mean * self.norm.weight / std
                self.cv = conv
            self.norm = nn.Identity()

        if self.gamma is not None:
            final_scale = self.gamma.data
            self.gamma = None
        else:
            final_scale = 1

        if self.act[1].use_bias and len(self.pwconv2) == 3:
            grn_bias = self.act[1].beta.data 
            self.act[1].__delattr__('beta')
            self.act[1].use_bias = False
            linear = self.pwconv2[0]
            grn_bias_projected_bias = (linear.weight.data @ grn_bias.view(-1, 1)).squeeze()
            bn = self.pwconv2[2]
            std = (bn.running_var + bn.eps).sqrt()
            new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
            new_linear.weight.data = linear.weight * (bn.weight / std * final_scale).view(-1, 1)
            linear_bias = 0 if linear.bias is None else linear.bias.data
            linear_bias += grn_bias_projected_bias
            new_linear.bias.data = (bn.bias + (linear_bias - bn.running_mean) * bn.weight / std) * final_scale
            self.pwconv2 = nn.Sequential(new_linear, self.pwconv2[1])

class S_UniRepLKNetBlock(nn.Module):
    def __init__(self,
                 c1,
                 c2,
                 shortcut=True,
                 s=1,
                 ):
        super().__init__()
        assert c1 == c2
        self.cv = RepVGGDW_ViT(c1)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.GELU()
        self.ffn = nn.Sequential(
    Conv(c1, c1 * 2, 1),
    nn.GELU(),
    Conv(c1 * 2, c1, 1, act=False)
)

        self.shortcut = nn.Sequential(Conv(c1, c2, k=1, s=s, act=False)) if s != 1 or c1 != c2 else nn.Identity()
    
    def forward(self,x):
        y = self.ffn(self.bn(self.cv(x)))
        return y+self.shortcut(x)


    
class Lark_Block(nn.Module):
    def __init__(self,c1,c2,shortcut=True,g=1,e=3/8,s=1,k=17):
        super().__init__()
                # 使用 nn.Sequential 来组织模块

        self.cv1 = L_UniRepLKNetBlock(c1,c2,e=e,shortcut=shortcut)
        self.cv2 = S_UniRepLKNetBlock(c1,c2,shortcut=shortcut)
                # 使用 nn.Sequential 来组织模块
    def forward(self,x):
        y = self.cv2(self.cv1(x))
        return y

        
class Smak_Block(nn.Module):
    def __init__(self,c1,c2,s=1,shortcut=True):
        super().__init__()
        self.cv1 = S_UniRepLKNetBlock(c1, c2, shortcut=shortcut)   # 第一个小核块
        self.cv2 = S_UniRepLKNetBlock(c2, c2, shortcut=shortcut)   # 第二个小核块
    def forward(self,x):
        y = self.cv2(self.cv1(x))
        return y

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia # 导入 kornia 库
import kornia.color as KC # 仅用于 COA.forward 中可能的初始灰度转换（如果需要备用方案）
import kornia.filters as KF # 用于 harris_response 和 dilation

# -----------------------------------------------------------------------------
# 函数：harris_kornia_optimized (处理单通道图像批次)
# -----------------------------------------------------------------------------
def harris_kornia_optimized(
    img_single_channel_batch: torch.Tensor, # 明确这是单通道输入 (N, 1, H, W)
    k: float = 0.04,
    threshold_ratio: float = 0.01, # 经验值，需要根据kornia的输出范围调整
    dilation_kernel_size: int = 3,
    gaussian_kernel_size: int = 5, # kornia harris_response 内部高斯模糊参数
    gaussian_sigma: float = 1.5,   # kornia harris_response 内部高斯模糊参数
    harris_window_size: int = 5    # kornia harris_response 内部窗口大小参数
    ) -> torch.Tensor:
    """
    Harris角点检测的 Kornia GPU优化版本，处理单通道图像批次。
    img_single_channel_batch: 输入的单通道图像批次，形状 (N, 1, H, W)。
    k: Harris 角点检测器自由参数。
    threshold_ratio: 用于对Harris响应进行阈值处理的比例因子（相对于最大响应）。
    dilation_kernel_size: 用于对检测到的角点进行膨胀的核大小 (0表示不膨胀)。
    gaussian_kernel_size: Kornia内部高斯模糊的核大小。
    gaussian_sigma: Kornia内部高斯模糊的sigma。
    harris_window_size: Kornia内部计算协方差矩阵的窗口大小。
    """
    if not (img_single_channel_batch.ndim == 4 and img_single_channel_batch.shape[1] == 1):
        raise ValueError(
            f"harris_kornia_optimized expects a single channel input (N, 1, H, W), "
            f"got {img_single_channel_batch.shape}"
        )
    device = img_single_channel_batch.device

    # 1. 使用 kornia.filters.harris_response 计算Harris响应
    harris_responses = KF.harris_response(
        img_single_channel_batch, # 直接使用单通道输入
        k=torch.tensor(k, device=device, dtype=img_single_channel_batch.dtype), # k 需要是tensor，且dtype匹配
        gaussian_kernel_size=(gaussian_kernel_size, gaussian_kernel_size),
        gaussian_sigma=(gaussian_sigma, gaussian_sigma),
        window_size=harris_window_size
    ) # 输出形状 (N, 1, H, W)

    # 2. 阈值化 R
    att_maps = torch.zeros_like(harris_responses) # 初始化输出 (N, 1, H, W)
    for i in range(harris_responses.shape[0]): # 遍历批次中的每个图像的响应
        R_single = harris_responses[i] # (1, H, W)
        
        current_max = R_single.max()
        if current_max > 1e-8: # 避免在非常小的响应上设置阈值
            threshold_val = threshold_ratio * current_max
            corner_mask = (R_single > threshold_val).to(img_single_channel_batch.dtype) # 确保dtype一致
        else: # 如果最大响应接近0，则没有角点
            corner_mask = torch.zeros_like(R_single, dtype=img_single_channel_batch.dtype)
            
        att_maps[i] = corner_mask

    # 3. 膨胀操作 (Dilation)
    if dilation_kernel_size > 0:
        dilation_kernel = torch.ones(dilation_kernel_size, dilation_kernel_size,
                                     device=device, dtype=img_single_channel_batch.dtype)
        # kornia.morphology.dilation 的输入是 (B, C, H, W) 和 kernel (H_k, W_k)
        att_maps = KF.dilation(att_maps, kernel=dilation_kernel, border_type='replicate')
        # dilation 后可能不是严格的0和1，如果需要严格二值，可以再次阈值
        # att_maps = (att_maps > 0).to(img_single_channel_batch.dtype)

    return att_maps # 返回形状 (N, 1, H, W) 的注意力图批次

# -----------------------------------------------------------------------------
# 模块：COA (Corner-Oriented Attention)
# -----------------------------------------------------------------------------
class COA(nn.Module):
    def __init__(self, 
                 channel: int, # 输入特征图的通道数 (例如，RGB图像为3)
                 harris_k: float = 0.04, 
                 harris_threshold_ratio: float = 0.01, # 经验值，可能需要为kornia的输出调整
                 harris_dilation_kernel: int = 3,      # 0 表示不进行膨胀
                 harris_gaussian_kernel: int = 5,    # harris_response参数
                 harris_gaussian_sigma: float = 1.5,   # harris_response参数
                 harris_window_sz: int = 5             # harris_response参数
                 ):
        super(COA, self).__init__()
        if channel <= 0:
            raise ValueError("Number of input channels must be positive.")
            
        # 1x1 卷积，用于特征变换
        self.conv_transform = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
        )
        
        self.num_input_channels = channel # 保存输入通道数，用于遍历
        
        # Harris角点检测相关的参数
        self.harris_k = harris_k
        self.harris_threshold_ratio = harris_threshold_ratio
        self.harris_dilation_kernel_size = harris_dilation_kernel
        self.harris_gaussian_kernel_size = harris_gaussian_kernel
        self.harris_gaussian_sigma = harris_gaussian_sigma
        self.harris_window_size = harris_window_sz

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x 形状 (N, C, H, W)
        if x.shape[1] != self.num_input_channels:
            raise ValueError(
                f"Input tensor has {x.shape[1]} channels, but COA was initialized with "
                f"{self.num_input_channels} channels."
            )

        # 1. 对输入特征x进行1x1卷积变换
        y = self.conv_transform(x) # y 形状 (N, C, H, W)

        # 2. 对每个颜色/输入通道独立计算Harris角点注意力图
        if self.num_input_channels > 0:
            per_channel_att_maps_list = []
            for i in range(self.num_input_channels):
                # 提取单个通道的数据
                single_channel_data = x[:, i:i+1, :, :] # 形状 (N, 1, H, W)
                
                # 为该通道计算Harris注意力图
                channel_specific_harris_att = harris_kornia_optimized(
                    single_channel_data,
                    k=self.harris_k,
                    threshold_ratio=self.harris_threshold_ratio,
                    dilation_kernel_size=self.harris_dilation_kernel_size,
                    gaussian_kernel_size=self.harris_gaussian_kernel_size,
                    gaussian_sigma=self.harris_gaussian_sigma,
                    harris_window_size=self.harris_window_size
                ) # 输出形状 (N, 1, H, W)
                per_channel_att_maps_list.append(channel_specific_harris_att)
            
            # 3. 合并来自各个通道的注意力图
            if per_channel_att_maps_list:
                # 将列表中的 (N, 1, H, W) 张量在通道维度(dim=1)上拼接起来
                # 得到 (N, C, H, W) 的多通道注意力图
                concatenated_att_maps = torch.cat(per_channel_att_maps_list, dim=1)
                
                # 在拼接后的通道维度上取最大值，得到一个最终的单通道空间注意力图
                # [0] 是因为 .max() 返回 (values, indices)
                final_spatial_att_map = concatenated_att_maps.max(dim=1, keepdim=True)[0] # 形状 (N, 1, H, W)
                # 其他合并策略也可以考虑，例如:
                # final_spatial_att_map = concatenated_att_maps.mean(dim=1, keepdim=True)
            else:
                # 如果由于某种原因列表为空（例如输入通道为0，尽管构造函数会阻止），
                # 创建一个不起增强作用的注意力图（全1）
                final_spatial_att_map = torch.ones_like(x[:, 0:1, :, :], dtype=x.dtype, device=x.device)
        else:
             # 如果输入通道为0
             final_spatial_att_map = torch.ones_like(x[:, 0:1, :, :], dtype=x.dtype, device=x.device)


        # 4. 将单通道空间注意力图应用于多通道特征y，并添加残差连接
        # PyTorch的广播机制: (N, 1, H, W) * (N, C, H, W) -> (N, C, H, W)
        out = final_spatial_att_map * y + x
        
        return out

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)
    
    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert(m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert(m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self

from timm.models.layers import SqueezeExcite

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

from torch.nn.modules.batchnorm import _BatchNorm

def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class RepViTBlock_ECA_Att(nn.Module):
    def __init__(self, c1, c2, use_se=True, stride=1, use_hs=True):
        """
        Args:
            c1: 输入通道数（自动从上一层获取）
            c2: 输出通道数
            stride: 步长，默认1
            use_se: 是否使用SE模块，默认True
            use_hs: 是否使用GELU激活，默认True
        """
        super(RepViTBlock_ECA_Att, self).__init__()
        assert stride in [1, 2]
        
        self.ega_att = EGA_att(dim=c1)
        
        self.identity = stride == 1 and c1 == c2
        hidden_dim = 2 * c1  # 隐藏层通道数固定为输入通道数的2倍

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(c1, c1, 3, stride, 1, groups=c1),
                SqueezeExcite(c1, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(c1, c2, 1, 1, 0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                    Conv2d_BN(c2, 2 * c2, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    Conv2d_BN(2 * c2, c2, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert(self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW_ViT(c1),
                SqueezeExcite(c1, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                    Conv2d_BN(c1, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    Conv2d_BN(hidden_dim, c2, 1, 1, 0, bn_weight_init=0),
                ))
        
    def forward(self, x):
        x = self.ega_att(x)
        return  self.channel_mixer(self.token_mixer(x))

class RepViTBlock_ECA(nn.Module):
    def __init__(self, c1, c2, use_se=True, stride=1, use_hs=True):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数
            stride: 步长
            use_se: 是否使用注意力模块 (现在它控制是否使用ECA)
            use_hs: 是否使用GELU激活
        """
        super(RepViTBlock_ECA, self).__init__()
        assert stride in [1, 2]
        
        
        self.identity = stride == 1 and c1 == c2
        hidden_dim = 2 * c1

        # 根据步长构建不同的 token_mixer 分支
        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(c1, c1, 3, stride, 1, groups=c1),
                # 修改点: 将 SqueezeExcite 替换为 ECA
                ECA(c1) if use_se else nn.Identity(),
                Conv2d_BN(c1, c2, 1, 1, 0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                    Conv2d_BN(c2, 2 * c2, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    Conv2d_BN(2 * c2, c2, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert(self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW_ViT(c1),
                # 修改点: 将 SqueezeExcite 替换为 ECA
                ECA(c1) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                    Conv2d_BN(c1, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    Conv2d_BN(hidden_dim, c2, 1, 1, 0, bn_weight_init=0),
                ))
        
    def forward(self, x):
        # forward 逻辑保持不变
        return self.channel_mixer(self.token_mixer(x))

import math


class ECA(nn.Module):
    """
    高效通道注意力 (Efficient Channel Attention - ECA) 模块。
    
    这是对您 `Edge_Emphasize` 模块中一维卷积注意力逻辑的独立封装。
    它的作用是接收一个特征图，然后学习每个通道的重要性权重，最后将权重乘回
    原始特征图上，从而实现对特征的动态增强。
    """
    def __init__(self, channel):
        """
        Args:
            channel (int): 输入特征图的通道数。
        """
        super(ECA, self).__init__()
        
        t = int(abs((math.log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.conv1d = nn.Conv1d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=k, 
            padding=(k - 1) // 2,  
            bias=False
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: 输入特征图，shape为 [B, C, H, W]
        """
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv1d(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

class RepVGGDW_ViT_LK(torch.nn.Module):
    """新版本：支持自定义卷积核大小 (Large Kernel)。"""
    def __init__(self, ed, kernel_size=7) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.conv = Conv2d_BN(ed, ed, kernel_size, 1, padding, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed, bias=True)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)
    
    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)
    
    @torch.no_grad()
    def fuse(self):
        # ... (与上一轮回复中修改后的大核版本融合逻辑一致)
        conv = self.conv.fuse()
        conv1 = self.conv1
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias if conv1.bias is not None else torch.zeros_like(conv_b)
        k_pad = self.kernel_size // 2
        conv1_w = torch.nn.functional.pad(conv1_w, [k_pad, k_pad, k_pad, k_pad])
        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [k_pad, k_pad, k_pad, k_pad])
        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b
        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / (bn.running_var + bn.eps)**0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv

class RepViTBlock_LK(nn.Module):
    """
    最终版本：
    - 无 Residual 辅助类，残差连接在 forward 方法中直接实现。
    """
    def __init__(self, c1, c2, kernel_size=7, use_se=True, stride=1, use_hs=True):
        super(RepViTBlock_LK, self).__init__()
        assert stride in [1, 2]
        
        # self.identity = stride == 1 and c1 == c2  # No longer needed after removing residuals
        
        padding = kernel_size // 2
        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(c1, c1, kernel_size, stride, padding, groups=c1),
                SqueezeExcite(c1, 0.25) if use_se else nn.Identity(),
            )
        else:
            self.token_mixer = nn.Sequential(
                RepVGGDW_ViT_LK(c1, kernel_size=kernel_size),
                SqueezeExcite(c1, 0.25) if use_se else nn.Identity(),
            )
        
        self.channel_mixer = nn.Sequential(
                Conv2d_BN(c1, 2 * c1, 1, 1, 0),
                nn.GELU(),
                Conv2d_BN(2 * c1, c2, 1, 1, 0, bn_weight_init=0),
            )

    def forward(self, x):
        # Removed all residual connections as requested.
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        return x

# class RepViTBlock_LK(nn.Module):
#     """
#     最终版本：
#     - 无 Residual 辅助类，残差连接在 forward 方法中直接实现。
#     """
#     def __init__(self, c1, c2, kernel_size=7, use_se=True, stride=1, use_hs=True):
#         super(RepViTBlock_LK, self).__init__()
#         assert stride in [1, 2]
        
#         self.identity = stride == 1 and c1 == c2
        
#         padding = kernel_size // 2
#         if stride == 2:
#             self.token_mixer = nn.Sequential(
#                 Conv2d_BN(c1, c1, kernel_size, stride, padding, groups=c1),
#                 SqueezeExcite(c1, 0.25) if use_se else nn.Identity(),
#             )
#         else:
#             self.token_mixer = nn.Sequential(
#                 RepVGGDW_ViT_LK(c1, kernel_size=kernel_size),
#                 SqueezeExcite(c1, 0.25) if use_se else nn.Identity(),
#             )
        
#         self.channel_mixer = nn.Sequential(
#                 Conv2d_BN(c1, 2 * c1, 1, 1, 0),
#                 nn.GELU(),
#                 Conv2d_BN(2 * c1, c2, 1, 1, 0, bn_weight_init=0),
#             )

#     def forward(self, x):
#         identity_input = x
        
#         if self.identity:
#             x_after_token_mixer = self.token_mixer(x) + x
#             x_after_channel_mixer = self.channel_mixer(x_after_token_mixer) + x_after_token_mixer
#             return x_after_channel_mixer 
#         else:
#             # 当 stride=2 或 c1!=c2 时，没有残差连接
#             x_after_token_mixer = self.token_mixer(x)
#             x_after_channel_mixer = self.channel_mixer(x_after_token_mixer)
#             return x_after_channel_mixer

    

# class RepViTBlock_LK(nn.Module):
#     """
#     修改后的版本：
#     - 实现了 CSP (Cross Stage Partial) 结构。
#     - 输入经过1x1卷积后分割成两路，一路深度处理，一路短路连接，最后融合。
#     """
#     def __init__(self, c1, c2, kernel_size=7, e=0.5, use_se=True):
#         """
#         Args:
#             c1 (int): 输入通道数。
#             c2 (int): 输出通道数。
#             kernel_size (int): Token Mixer 中大核卷积的核大小。
#             e (float): 分割和扩展比例，用于确定中间处理部分的通道数。
#             use_se (bool): 是否在 Token Mixer 中使用 Squeeze-and-Excite 注意力模块。
#         """
#         super(RepViTBlock_LK, self).__init__()
        
#         # 计算每个分割分支的中间通道数
#         self.c_ = int(c2 * e)  # intermediate channels for the deep branch

#         # 1. 初始 1x1 卷积，用于生成可供分割的特征，输出通道数为 c_ * 2
#         self.initial_conv = Conv2d_BN(c1, self.c_ * 2, 1, 1, 0)
        
#         # 2. 第一个分割分支的主处理路径（深度路径）
#         # 该路径接收 c_ 通道，并输出 c_ 通道
#         self.token_mixer = nn.Sequential(
#             RepVGGDW_ViT_LK(self.c_, kernel_size=kernel_size),
#             SqueezeExcite(self.c_, 0.25) if use_se else nn.Identity(),
#         )
        
#         # channel_mixer 接收 c_ 通道，内部先扩展再压缩回 c_ 通道
#         self.channel_mixer = nn.Sequential(
#                 Conv2d_BN(self.c_, 2 * self.c_, 1, 1, 0),
#                 nn.GELU(),
#                 Conv2d_BN(2 * self.c_, self.c_, 1, 1, 0, bn_weight_init=0),
#             )
        
#         # 3. 最终的 1x1 卷积，用于融合拼接后的特征
#         # 它接收拼接后的张量，尺寸为 (c_ + c_) = 2 * c_，并输出最终的 c2 通道数。
#         self.fusion_conv = Conv2d_BN(self.c_ * 2, c2, 1, 1, 0)

#     def forward(self, x):
#         """
#         实现 CSP 结构的前向传播。
#         """
#         # 应用初始卷积
#         x_initial = self.initial_conv(x)
        
#         # 在通道维度上将张量分割成两部分
#         part1, part2 = x_initial.chunk(2, dim=1)
        
#         # 对 part1 应用深度处理路径
#         x_token_mixed = self.token_mixer(part1)
#         # 应用带有内部残差连接的 channel mixer
#         part1_processed = self.channel_mixer(x_token_mixed) + x_token_mixed

#         # 将处理过的 part1 和未处理的 part2 进行拼接
#         x_concatenated = torch.cat((part1_processed, part2), dim=1)
        
#         # 应用最终的融合卷积
#         output = self.fusion_conv(x_concatenated)
        
#         return output





class EdgeConvSobelX(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeConvSobelX, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
        self.scale = nn.Parameter(torch.randn(out_channels, 1, 1, 1) * 1e-3)
        self.bias = nn.Parameter(torch.randn(out_channels))
        template = torch.zeros(out_channels, 1, 3, 3)
        for i in range(out_channels):
            template[i, 0, 0, 0] = 1.0; template[i, 0, 1, 0] = 2.0; template[i, 0, 2, 0] = 1.0
            template[i, 0, 0, 2] = -1.0; template[i, 0, 1, 2] = -2.0; template[i, 0, 2, 2] = -1.0
        self.template = nn.Parameter(template, requires_grad=False)
    def forward(self, x):
        y0 = self.conv1x1(x)
        return F.conv2d(y0, self.scale * self.template, self.bias, 1, 1, groups=y0.shape[1])

class EdgeConvSobelY(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeConvSobelY, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
        self.scale = nn.Parameter(torch.randn(out_channels, 1, 1, 1) * 1e-3)
        self.bias = nn.Parameter(torch.randn(out_channels))
        template = torch.zeros(out_channels, 1, 3, 3)
        for i in range(out_channels):
            template[i, 0, 0, 0] = 1.0; template[i, 0, 0, 1] = 2.0; template[i, 0, 0, 2] = 1.0
            template[i, 0, 2, 0] = -1.0; template[i, 0, 2, 1] = -2.0; template[i, 0, 2, 2] = -1.0
        self.template = nn.Parameter(template, requires_grad=False)
    def forward(self, x):
        y0 = self.conv1x1(x)
        return F.conv2d(y0, self.scale * self.template, self.bias, 1, 1, groups=y0.shape[1])

class MAFM_Fusion(nn.Module):
    """
    Multi-scale Attentional Feature Fusion Module.
    Takes two parallel feature branches and fuses them using a dynamic, 
    softmax-based attention mechanism.
    """
    def __init__(self, channels, r=2, L=32):
        super(MAFM_Fusion, self).__init__()
        # Calculate the intermediate dimension 'd'
        d = max(int(channels / r), L)

        # FC layer for squeezing global information
        self.fc = nn.Linear(channels, d)
        
        # Two independent FC layers to generate attention vectors for each branch
        self.fcs = nn.ModuleList([nn.Linear(d, channels) for _ in range(2)])
        
        # Softmax to force competition between branches
        self.softmax = nn.Softmax(dim=1)

    def forward(self, branch1_features, branch2_features):
        # branch1_features: dw_out [B, C, H, W]
        # branch2_features: edge_out [B, C, H, W]

        # 1. Package the two branches into a single tensor
        # New shape: [B, 2, C, H, W] where dim=1 represents the branches
        features_packed = torch.cat([
            branch1_features.unsqueeze(1), 
            branch2_features.unsqueeze(1)
        ], dim=1)

        # 2. Generate global information descriptor by fusing and squeezing
        # Summing across branches and then applying global average pooling
        global_descriptor = torch.sum(features_packed, dim=1).mean((2, 3)) # Shape: [B, C]

        # 3. Compress information
        global_descriptor_compressed = self.fc(global_descriptor) # Shape: [B, d]

        # 4. Generate attention vectors for each branch independently
        attention_vectors = [fc(global_descriptor_compressed) for fc in self.fcs]
        attention_vectors_packed = torch.stack(attention_vectors, dim=1) # Shape: [B, 2, C]

        # 5. Apply softmax competition
        attention_weights = self.softmax(attention_vectors_packed)
        # Reshape for broadcasting: [B, 2, C, 1, 1]
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)

        # 6. Apply weights and fuse by summation
        # Element-wise multiplication followed by summing across the branch dimension
        fused_output = (features_packed * attention_weights).sum(dim=1)

        return fused_output

class BasicBlock(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            act = nn.GELU()
            m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResidualBlock(nn.Module):
    """
    A standard residual block that includes BatchNorm layers.
    Structure: Conv-BN-Act -> Conv-BN -> + -> Act
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False, act=nn.PReLU()):
        super(ResidualBlock, self).__init__()
        
        # We use two BasicBlocks for the main path
        self.block1 = BasicBlock(in_channels, out_channels, kernel_size, stride=stride, bias=bias, bn=True, act=act)
        # The second block's activation is applied after the residual connection
        self.block2 = BasicBlock(out_channels, out_channels, kernel_size, stride=1, bias=bias, bn=True, act=None)

        # Shortcut for residual connection if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels)
            )
            
        # Final activation after adding the shortcut
        self.final_act = act if act is not None else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.block1(x)
        out = self.block2(out)
        return self.final_act(identity + out)

class RepViTBlock_edge(nn.Module):
    """
    轻量化版本：
    - 实现了 CSP (Cross Stage Partial) 结构。
    - 输入经过1x1卷积后分割成两路，一路进行包含边缘和主干分支的深度处理，
      另一路作为短路连接，最后将两路结果融合。
    - 主干分支使用了一个标准的 ResidualBlock。
    """
    def __init__(self, c1, c2, use_se=True, stride=1, use_hs=True, e=0.5):
        """
        Args:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            use_se (bool): 是否在特征融合时使用 MAFM_Fusion 模块。
            e (float): 分割和扩展比例，用于确定中间处理部分的通道数。
        """
        super(RepViTBlock_edge, self).__init__()
        
        # 计算深度处理分支的中间通道数
        self.c_ = int(c2 * e)

        # 1. 初始 1x1 卷积，生成可供分割的特征
        self.initial_conv = Conv2d_BN(c1, self.c_ * 2, 1, 1, 0)

        # 2. 深度处理路径 (操作于 self.c_ 通道上)
        # 2a. 边缘分支 (无变化)
        self.edge_sobel_x = EdgeConvSobelX(self.c_, self.c_)
        self.edge_sobel_y = EdgeConvSobelY(self.c_, self.c_)
        self.conv_reduce = Conv2d_BN(self.c_ * 2, self.c_, ks=1)
        self.sa = SAM()
        
        # 2b. 主干分支与融合逻辑 (核心改动)
        self.use_se = use_se
        if self.use_se:
            self.fusion = MAFM_Fusion(self.c_)
        
        # 将 dw_conv 替换为 ResidualBlock
        # 并将变量重命名为 conv_branch 以反映其类型变化
        self.conv_branch = ResidualBlock(self.c_, self.c_, act=nn.GELU())

        # 2c. Channel Mixer (无变化)
        hidden_dim = 2 * self.c_
        self.channel_mixer = nn.Sequential(
            Conv2d_BN(self.c_, hidden_dim, ks=1),
            nn.GELU(),
            Conv2d_BN(hidden_dim, self.c_, ks=1, bn_weight_init=0),
        )

        # 3. 最终的 1x1 卷积，用于融合拼接后的特征 (无变化)
        self.fusion_conv = Conv2d_BN(self.c_ * 2, c2, 1, 1, 0)
    
    def forward(self, x):
        # 1. 初始卷积和分割
        x_initial = self.initial_conv(x)
        part1, part2 = x_initial.chunk(2, dim=1)

        # 2. 对 part1 进行深度处理
        # 2a. 边缘分支
        edge_x = self.edge_sobel_x(part1)
        edge_y = self.edge_sobel_y(part1)
        edge_cat = torch.cat([edge_x, edge_y], dim=1)
        edge_reduced = self.conv_reduce(edge_cat)
        edge_out = self.sa(edge_reduced) * edge_reduced
        
        # 2b. 主干卷积分支 (核心改动)
        conv_out = self.conv_branch(part1)
        
        # 2c. 动态特征融合
        if self.use_se:
            fused_features = self.fusion(conv_out, edge_out)
        else:
            fused_features = conv_out + edge_out
        
        # 2d. Channel Mixer 和残差连接
        part1_processed = fused_features + self.channel_mixer(fused_features)

        # 3. 拼接与融合
        x_concatenated = torch.cat((part1_processed, part2), dim=1)
        output = self.fusion_conv(x_concatenated)
        
        return output


class RepViTBlock(nn.Module):
    def __init__(self, c1, c2, use_se=True, stride=1, use_hs=True):
        """
        Args:
            c1: 输入通道数（自动从上一层获取）
            c2: 输出通道数
            stride: 步长，默认1
            use_se: 是否使用SE模块，默认True
            use_hs: 是否使用GELU激活，默认True
        """
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]
        
        self.identity = stride == 1 and c1 == c2
        hidden_dim = 2 * c1  # 隐藏层通道数固定为输入通道数的2倍

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(c1, c1, 3, stride, 1, groups=c1),
                SqueezeExcite(c1, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(c1, c2, 1, 1, 0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                    Conv2d_BN(c2, 2 * c2, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    Conv2d_BN(2 * c2, c2, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert(self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW_ViT(c1),
                SqueezeExcite(c1, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                    Conv2d_BN(c1, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    Conv2d_BN(hidden_dim, c2, 1, 1, 0, bn_weight_init=0),
                ))
        
    def forward(self, x):
        return  self.channel_mixer(self.token_mixer(x))




import torch.nn as nn
import torch
from torch.nn import functional as F

class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
      
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)


    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        x = torch.sigmoid(x) * residual #
        
        return x


class Att(nn.Module):
    def __init__(self, channels,shape, out_channels=None, no_spatial=True):
        super(Att, self).__init__()
        self.Channel_Att = Channel_Att(channels)
  
    def forward(self, x):
        x_out1=self.Channel_Att(x)
 
        return x_out1  

class DenseRepViTLayer(nn.Module):
    def __init__(self, c1, c2,use_se=True,stride=1,use_hs=True):
        super().__init__()
        self.denseblock = nn.Sequential(
            RepViTBlock(c1,c2,use_se=use_se,stride=stride,use_hs=use_hs),
            RepViTBlock(c2,c2,use_se=use_se,stride=stride,use_hs=use_hs)
        )

    
    def forward(self,x: Tensor) -> Tensor:
        y = self.denseblock(x)
        return y


class DenseRepViTLayer_Edge(nn.Module):
    def __init__(self, c1, c2, use_se=True, stride=1, use_hs=True,e=0.5):
        super().__init__()
        
        # 构建一个包含不同类型块的 Sequential 模块
        self.denseblock = nn.Sequential(
            # 第一个块：使用我们功能更强大的边缘感知版本
            RepViTBlock_edge(c1, c2, use_se=use_se, stride=stride, use_hs=use_hs, e=e),
            
            # 第二个块：使用原始的、更轻量的版本
            RepViTBlock(c2, c2, use_se=use_se, stride=stride, use_hs=use_hs)
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.denseblock(x)
        return y

class DenseRepViTLayer_LK(nn.Module):
    # 新增 kernel_size 和 reduction_ratio 参数
    def __init__(self, c1, c2, use_se=True, stride=1, use_hs=True, kernel_size=7):
        super().__init__()
        # 将 RepViTBlock 替换为 RepViTBlock_LK，并传递所有参数
        self.denseblock = nn.Sequential(
            RepViTBlock(c1,c2,use_se=use_se,stride=stride,use_hs=use_hs),
            RepViTBlock_LK(c2, c2, kernel_size, use_se, stride, use_hs=use_hs)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.denseblock(x)
    

class CSP_DenseRepViTBlock_LK(nn.Module):
    # __init__ 需要包含 Hybrid Layer 所需的所有参数
    def __init__(self, c1, c2, num_layers, use_se, csp_frac=0.5, stride=1, use_hs=True, kernel_size=7):
        super().__init__()
        assert c1 == c2, "Input and output channels must be the same"
        self.num_layers = num_layers

        self.c_heavy = int(c1 * csp_frac)
        self.c_light = c1 - self.c_heavy
        self.conv_shortcut = Conv(self.c_light, self.c_light, 1, 1)

        self.layers = nn.ModuleList()
        # 只有第一个 Hybrid Layer 可能有 stride=2
        if num_layers > 0:
            self.layers.append(
                DenseRepViTLayer_LK(
                    self.c_heavy, self.c_heavy, use_se, stride, use_hs, kernel_size
                )
            )
        # 后续所有 Hybrid Layer 的 stride 都必须是 1
        for _ in range(num_layers - 1):
            self.layers.append(
                DenseRepViTLayer_LK(
                    self.c_heavy, self.c_heavy, use_se, stride, use_hs, kernel_size
                )
            )
            
        # 密集连接和融合部分的代码保持不变
        self.reduction_modules = nn.ModuleList()
        for i in range(num_layers - 1):
            in_channels_for_reduction = (i + 2) * self.c_heavy
            self.reduction_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_for_reduction, self.c_heavy, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.c_heavy),
                    nn.GELU()
                )
            )

        final_in_channels_heavy = (num_layers + 1) * self.c_heavy
        self.final_reduction = nn.Sequential(
            nn.Conv2d(final_in_channels_heavy, self.c_heavy, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c_heavy),
            nn.GELU()
        )

        self.final_fusion = Conv(self.c_light + self.c_heavy, c2, 1, 1)
        self.final_conv = Conv(c2, c2, 1, 1)

    def forward(self, input_features: Tensor) -> Tensor:
        # forward 方法的逻辑完全不需要改变
        x_light, x_heavy = input_features.split([self.c_light, self.c_heavy], dim=1)
        out_light = self.conv_shortcut(x_light)

        features = [x_heavy]
        current_input_to_dense_layer = x_heavy
        for i in range(self.num_layers):
            if i > 0:
                concat_for_shuffle = torch.cat(features, dim=1)
                shuffled_features = channel_shuffle(concat_for_shuffle, groups=len(features))
                current_input_to_dense_layer = self.reduction_modules[i - 1](shuffled_features)
            layer_output = self.layers[i](current_input_to_dense_layer)
            features.append(layer_output)
        final_block_output = torch.cat(features, dim=1)
        shuffled_final_output = channel_shuffle(final_block_output, groups=len(features))
        out_heavy = self.final_reduction(shuffled_final_output)
        
        out = torch.cat((out_light, out_heavy), dim=1)
        return self.final_conv(self.final_fusion(out))


class CSP_DenseRepViTBlock_lK(nn.Module):
    # 在 __init__ 中添加 kernel_size 和 reduction_ratio 参数
    def __init__(self, c1, c2, num_layers, use_se, csp_frac=0.5, stride=1, use_hs=True, kernel_size=7):
        super().__init__()
        assert c1 == c2, "Input and output channels must be the same for CSP DenseRepViTBlock_"
        self.num_layers = num_layers

        self.c_heavy = int(c1 * csp_frac)
        self.c_light = c1 - self.c_heavy
        self.conv_shortcut = Conv(self.c_light, self.c_light, 1, 1)

        self.layers = nn.ModuleList()
        if num_layers > 0:
            # 将 RepViTBlock 替换为 RepViTBlock_LK
            # 并传递新增的参数 kernel_size 和 reduction_ratio
            self.layers.append(
                RepViTBlock_LK(
                    self.c_heavy, self.c_heavy, 
                    kernel_size=kernel_size, 
                    use_se=use_se, 
                    stride=stride, 
                    use_hs=use_hs
                )
            )
        for _ in range(num_layers - 1):
            # 对后续层也进行同样的操作
            self.layers.append(
                RepViTBlock_LK(
                    self.c_heavy, self.c_heavy, 
                    kernel_size=kernel_size, 
                    use_se=use_se, 
                    stride=1, 
                    use_hs=use_hs
                )
            )

        self.reduction_modules = nn.ModuleList()
        for i in range(num_layers - 1):
            in_channels_for_reduction = (i + 2) * self.c_heavy
            self.reduction_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_for_reduction, self.c_heavy, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.c_heavy),
                    nn.GELU()
                )
            )

        final_in_channels_heavy = (num_layers + 1) * self.c_heavy
        self.final_reduction = nn.Sequential(
            nn.Conv2d(final_in_channels_heavy, self.c_heavy, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c_heavy),
            nn.GELU()
        )

        self.final_fusion = Conv(self.c_light + self.c_heavy, c2, 1, 1)
        self.final_conv = Conv(c2, c2, 1, 1)

    # forward 方法不需要改变
    def forward(self, input_features: Tensor) -> Tensor:
        x_light, x_heavy = input_features.split([self.c_light, self.c_heavy], dim=1)
        out_light = self.conv_shortcut(x_light)

        features = [x_heavy]
        current_input_to_dense_layer = x_heavy

        for i in range(self.num_layers):
            if i > 0:
                concat_for_shuffle = torch.cat(features, dim=1)
                num_parts_for_shuffle = len(features)
                shuffled_features = channel_shuffle(concat_for_shuffle, groups=num_parts_for_shuffle)
                current_input_to_dense_layer = self.reduction_modules[i - 1](shuffled_features)

            layer_output = self.layers[i](current_input_to_dense_layer)
            features.append(layer_output)

        final_block_output = torch.cat(features, dim=1)
        shuffled_final_output = channel_shuffle(final_block_output, groups=len(features))
        out_heavy = self.final_reduction(shuffled_final_output)

        out = torch.cat((out_light, out_heavy), dim=1)
        return self.final_conv(self.final_fusion(out))


class DenseRepViTBlock_Edge(nn.Module):
    def __init__(self, c1, c2, constant, num_layers,use_se,stride=1,use_hs=True,e=0.5):
        super().__init__()
        assert c1 == c2
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.constant = constant

        for _ in range(num_layers):
            self.layers.append (
                DenseRepViTLayer_Edge(c1, c2, use_se=use_se, stride=stride, use_hs=use_hs,e=e
                )
            )
        self.reduction_modules = nn.ModuleList()
        for i in range(num_layers-1):
            num_tensors_to_concatenate = i + 2
            in_channels_for_reduction = num_tensors_to_concatenate * constant

            self.reduction_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_for_reduction, self.constant, kernel_size=1, bias=False),
                    nn.BatchNorm2d(constant),
                    nn.GELU() # 或者 nn.ReLU(inplace=True)
                )
            )
        final_in_channels = (num_layers + 1) * constant
        self.final_reduction = nn.Sequential(
            nn.Conv2d(final_in_channels, self.constant, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.constant),
            nn.GELU()
        )


    def forward(self, input_features: Tensor) -> Tensor:

        features = [input_features]
        current_input_to_dense_layer = input_features

        for i in range(self.num_layers):
            if i > 0:
               concat_for_shuffle = torch.cat(features, dim=1)
               num_parts_for_shuffle = len(features)
               shuffled_features = channel_shuffle(concat_for_shuffle, groups=num_parts_for_shuffle)

               current_input_to_dense_layer = self.reduction_modules[i-1](shuffled_features)

            
            layer_output = self.layers[i](current_input_to_dense_layer)
            features.append(layer_output)
        
        final_block_output = torch.cat(features, dim=1)
        shuffled_final_output = channel_shuffle(final_block_output, groups=len(features))
        final_output = self.final_reduction(shuffled_final_output)

        return final_output

class DenseRepViTBlock(nn.Module):
    def __init__(self, c1, c2, constant, num_layers,use_se,stride=1,use_hs=True):
        super().__init__()
        assert c1 == c2
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.constant = constant

        for _ in range(num_layers):
            self.layers.append (
                DenseRepViTLayer(c1, c2, use_se=use_se, stride=stride, use_hs=use_hs
                )
            )
        self.reduction_modules = nn.ModuleList()
        for i in range(num_layers-1):
            num_tensors_to_concatenate = i + 2
            in_channels_for_reduction = num_tensors_to_concatenate * constant

            self.reduction_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_for_reduction, self.constant, kernel_size=1, bias=False),
                    nn.BatchNorm2d(constant),
                    nn.GELU() # 或者 nn.ReLU(inplace=True)
                )
            )
        final_in_channels = (num_layers + 1) * constant
        self.final_reduction = nn.Sequential(
            nn.Conv2d(final_in_channels, self.constant, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.constant),
            nn.GELU()
        )


    def forward(self, input_features: Tensor) -> Tensor:

        features = [input_features]
        current_input_to_dense_layer = input_features

        for i in range(self.num_layers):
            if i > 0:
               concat_for_shuffle = torch.cat(features, dim=1)
               num_parts_for_shuffle = len(features)
               shuffled_features = channel_shuffle(concat_for_shuffle, groups=num_parts_for_shuffle)

               current_input_to_dense_layer = self.reduction_modules[i-1](shuffled_features)

            
            layer_output = self.layers[i](current_input_to_dense_layer)
            features.append(layer_output)
        
        final_block_output = torch.cat(features, dim=1)
        shuffled_final_output = channel_shuffle(final_block_output, groups=len(features))
        final_output = self.final_reduction(shuffled_final_output)

        return final_output

class CSP_DenseRepViTBlock_Edge(nn.Module):
    def __init__(self, c1, c2, constant, num_layers,use_se, csp_frac=0.5,stride=1,use_hs=True,e=0.5):
        super().__init__()
        assert c1 == c2, "Input and output channels must be the same for CSP DenseRepViTBlock"
        self.num_layers = num_layers
        self.constant = constant  # Keep original parameter assignment

        # --- CSP Architecture Setup ---
        # 1. Store channel splits as attributes using the new `csp_frac` parameter
        self.c_heavy = int(c1 * csp_frac)
        self.c_light = c1 - self.c_heavy

        # 2. Define the light path (a simple 1x1 convolution)
        self.conv_shortcut = Conv(self.c_light, self.c_light, 1, 1)

        # 3. Define the heavy path (the original dense logic, adapted for c_heavy channels)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                DenseRepViTLayer_Edge(self.c_heavy, self.c_heavy, use_se=use_se, stride=stride, use_hs=use_hs,e=e)
            )

        self.reduction_modules = nn.ModuleList()
        for i in range(num_layers - 1):
            in_channels_for_reduction = (i + 2) * self.c_heavy
            self.reduction_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_for_reduction, self.c_heavy, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.c_heavy),
                    nn.GELU()
                )
            )

        # Final reduction for the heavy path, outputting c_heavy channels
        final_in_channels_heavy = (num_layers + 1) * self.c_heavy
        self.final_reduction = nn.Sequential(
            nn.Conv2d(final_in_channels_heavy, self.c_heavy, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c_heavy),
            nn.GELU()
        )

        # 4. Final fusion layer to combine light and heavy paths
        self.final_fusion = Conv(self.c_light + self.c_heavy, c2, 1, 1)
        self.final_conv = Conv(c2, c2, 1, 1)


    def forward(self, input_features: Tensor) -> Tensor:
        # Split input for CSP using stored attributes
        x_light, x_heavy = input_features.split([self.c_light, self.c_heavy], dim=1)

        # --- Process Light Path ---
        out_light = self.conv_shortcut(x_light)

        # --- Process Heavy Path ---
        features = [x_heavy]
        current_input_to_dense_layer = x_heavy

        for i in range(self.num_layers):
            if i > 0:
                concat_for_shuffle = torch.cat(features, dim=1)
                num_parts_for_shuffle = len(features)
                shuffled_features = channel_shuffle(concat_for_shuffle, groups=num_parts_for_shuffle)
                current_input_to_dense_layer = self.reduction_modules[i - 1](shuffled_features)

            layer_output = self.layers[i](current_input_to_dense_layer)
            features.append(layer_output)

        final_block_output = torch.cat(features, dim=1)
        shuffled_final_output = channel_shuffle(final_block_output, groups=len(features))
        out_heavy = self.final_reduction(shuffled_final_output)

        # --- Concatenate and Fuse ---
        out = torch.cat((out_light, out_heavy), dim=1)
        return self.final_conv(self.final_fusion(out))


class DenseRepViTBlock_edge(nn.Module):
    def __init__(self, c1, c2, constant, num_layers,use_se,stride=1,use_hs=True,e=0.5):
        super().__init__()
        assert c1 == c2
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.constant = constant

        for _ in range(num_layers):
            self.layers.append (
                RepViTBlock_edge(c1, c2, use_se=use_se, stride=stride, use_hs=use_hs, e=e
                )
            )
        self.reduction_modules = nn.ModuleList()
        for i in range(num_layers-1):
            num_tensors_to_concatenate = i + 2
            in_channels_for_reduction = num_tensors_to_concatenate * constant

            self.reduction_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_for_reduction, self.constant, kernel_size=1, bias=False),
                    nn.BatchNorm2d(constant),
                    nn.GELU() # 或者 nn.ReLU(inplace=True)
                )
            )
        final_in_channels = (num_layers + 1) * constant
        self.final_reduction = nn.Sequential(
            nn.Conv2d(final_in_channels, self.constant, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.constant),
            nn.GELU()
        )


    def forward(self, input_features: Tensor) -> Tensor:

        features = [input_features]
        current_input_to_dense_layer = input_features

        for i in range(self.num_layers):
            if i > 0:
               concat_for_shuffle = torch.cat(features, dim=1)
               num_parts_for_shuffle = len(features)
               shuffled_features = channel_shuffle(concat_for_shuffle, groups=num_parts_for_shuffle)

               current_input_to_dense_layer = self.reduction_modules[i-1](shuffled_features)

            
            layer_output = self.layers[i](current_input_to_dense_layer)
            features.append(layer_output)
        
        final_block_output = torch.cat(features, dim=1)
        shuffled_final_output = channel_shuffle(final_block_output, groups=len(features))
        final_output = self.final_reduction(shuffled_final_output)
        
        return final_output

class CSP_DenseRepViTBlock_edge(nn.Module):
    # 我们只修改 RepViTBlock 的调用，其他所有逻辑保持不变
    def __init__(self, c1, c2, constant, num_layers, use_se, csp_frac=0.5, stride=1, use_hs=True):
        super().__init__()
        assert c1 == c2, "Input and output channels must be the same for this block"
        self.num_layers = num_layers
        self.constant = constant

        self.c_heavy = int(c1 * csp_frac)
        self.c_light = c1 - self.c_heavy

        self.conv_shortcut = Conv(self.c_light, self.c_light, 1, 1)

        self.layers = nn.ModuleList()
        if num_layers > 0:
            # --- 核心改动点 1 ---
            self.layers.append(
                RepViTBlock_edge(self.c_heavy, self.c_heavy, use_se=use_se, stride=stride, use_hs=use_hs)
            )
        for _ in range(num_layers - 1):
            # --- 核心改动点 2 ---
            self.layers.append(
                RepViTBlock_edge(self.c_heavy, self.c_heavy, use_se=use_se, stride=1, use_hs=use_hs)
            )
        
        # --- 其余所有代码与您提供的版本完全相同 ---
        self.reduction_modules = nn.ModuleList()
        for i in range(num_layers - 1):
            in_channels_for_reduction = (i + 2) * self.c_heavy
            self.reduction_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_for_reduction, self.c_heavy, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.c_heavy),
                    nn.GELU()
                )
            )

        final_in_channels_heavy = (num_layers + 1) * self.c_heavy
        self.final_reduction = nn.Sequential(
            nn.Conv2d(final_in_channels_heavy, self.c_heavy, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c_heavy),
            nn.GELU()
        )

        self.final_fusion = Conv(self.c_light + self.c_heavy, c2, 1, 1)
        self.final_conv = Conv(c2, c2, 1, 1)

    def forward(self, input_features: Tensor) -> Tensor:
        # forward 方法的逻辑也完全不需要改变
        x_light, x_heavy = input_features.split([self.c_light, self.c_heavy], dim=1)
        out_light = self.conv_shortcut(x_light)

        features = [x_heavy]
        current_input_to_dense_layer = x_heavy
        for i in range(self.num_layers):
            if i > 0:
                concat_for_shuffle = torch.cat(features, dim=1)
                shuffled_features = channel_shuffle(concat_for_shuffle, groups=len(features))
                current_input_to_dense_layer = self.reduction_modules[i - 1](shuffled_features)
            
            layer_output = self.layers[i](current_input_to_dense_layer)
            features.append(layer_output)

        final_block_output = torch.cat(features, dim=1)
        shuffled_final_output = channel_shuffle(final_block_output, groups=len(features))
        out_heavy = self.final_reduction(shuffled_final_output)

        out = torch.cat((out_light, out_heavy), dim=1)
        return self.final_conv(self.final_fusion(out))


class CSP_DenseRepViTBlock(nn.Module):
    def __init__(self, c1, c2, num_layers,use_se, csp_frac=0.5,stride=1,use_hs=True):
        super().__init__()
        assert c1 == c2, "Input and output channels must be the same for CSP DenseRepViTBlock"
        self.num_layers = num_layers

        # --- CSP Architecture Setup ---
        # 1. Store channel splits as attributes using the new `csp_frac` parameter
        self.c_heavy = int(c1 * csp_frac)
        self.c_light = c1 - self.c_heavy

        # 2. Define the light path (a simple 1x1 convolution)
        self.conv_shortcut = Conv(self.c_light, self.c_light, 1, 1)

        # 3. Define the heavy path (the original dense logic, adapted for c_heavy channels)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                DenseRepViTLayer(self.c_heavy, self.c_heavy, use_se=use_se, stride=stride, use_hs=use_hs)
            )

        self.reduction_modules = nn.ModuleList()
        for i in range(num_layers - 1):
            in_channels_for_reduction = (i + 2) * self.c_heavy
            self.reduction_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_for_reduction, self.c_heavy, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.c_heavy),
                    nn.GELU()
                )
            )

        # Final reduction for the heavy path, outputting c_heavy channels
        final_in_channels_heavy = (num_layers + 1) * self.c_heavy
        self.final_reduction = nn.Sequential(
            nn.Conv2d(final_in_channels_heavy, self.c_heavy, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c_heavy),
            nn.GELU()
        )

        # 4. Final fusion layer to combine light and heavy paths
        self.final_fusion = Conv(self.c_light + self.c_heavy, c2, 1, 1)
        self.final_conv = Conv(c2, c2, 1, 1)


    def forward(self, input_features: Tensor) -> Tensor:
        # Split input for CSP using stored attributes
        x_light, x_heavy = input_features.split([self.c_light, self.c_heavy], dim=1)

        # --- Process Light Path ---
        out_light = self.conv_shortcut(x_light)

        # --- Process Heavy Path ---
        features = [x_heavy]
        current_input_to_dense_layer = x_heavy

        for i in range(self.num_layers):
            if i > 0:
                concat_for_shuffle = torch.cat(features, dim=1)
                num_parts_for_shuffle = len(features)
                shuffled_features = channel_shuffle(concat_for_shuffle, groups=num_parts_for_shuffle)
                current_input_to_dense_layer = self.reduction_modules[i - 1](shuffled_features)

            layer_output = self.layers[i](current_input_to_dense_layer)
            features.append(layer_output)

        final_block_output = torch.cat(features, dim=1)
        shuffled_final_output = channel_shuffle(final_block_output, groups=len(features))
        out_heavy = self.final_reduction(shuffled_final_output)

        # --- Concatenate and Fuse ---
        out = torch.cat((out_light, out_heavy), dim=1)
        return self.final_conv(self.final_fusion(out))


class CSP_DenseRepViTBlock_(nn.Module):
    def __init__(self, c1, c2, num_layers,use_se, csp_frac=0.5,stride=1,use_hs=True):
        super().__init__()
        assert c1 == c2, "Input and output channels must be the same for CSP DenseRepViTBlock_"
        self.num_layers = num_layers

        # --- CSP Architecture Setup ---
        # 1. Store channel splits as attributes using the new `csp_frac` parameter
        self.c_heavy = int(c1 * csp_frac)
        self.c_light = c1 - self.c_heavy

        # 2. Define the light path (a simple 1x1 convolution)
        self.conv_shortcut = Conv(self.c_light, self.c_light, 1, 1)

        # 3. Define the heavy path (the original dense logic, adapted for c_heavy channels)
        self.layers = nn.ModuleList()
        # The first layer of the heavy path might have a stride
        if num_layers > 0:
            self.layers.append(
                RepViTBlock(self.c_heavy, self.c_heavy, use_se=use_se, stride=stride, use_hs=use_hs)
            )
        # Subsequent layers have stride=1
        for _ in range(num_layers - 1):
            self.layers.append(
                RepViTBlock(self.c_heavy, self.c_heavy, use_se=use_se, stride=1, use_hs=use_hs)
            )

        self.reduction_modules = nn.ModuleList()
        for i in range(num_layers - 1):
            # All concatenated features in the heavy path have c_heavy channels
            in_channels_for_reduction = (i + 2) * self.c_heavy
            self.reduction_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_for_reduction, self.c_heavy, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.c_heavy),
                    nn.GELU()
                )
            )

        # Final reduction for the heavy path, outputting c_heavy channels
        final_in_channels_heavy = (num_layers + 1) * self.c_heavy
        self.final_reduction = nn.Sequential(
            nn.Conv2d(final_in_channels_heavy, self.c_heavy, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c_heavy),
            nn.GELU()
        )

        # 4. Final fusion layer to combine light and heavy paths
        self.final_fusion = Conv(self.c_light + self.c_heavy, c2, 1, 1)
        self.final_conv = Conv(c2, c2, 1, 1)


    def forward(self, input_features: Tensor) -> Tensor:
        # Split input for CSP using attributes stored during initialization.
        x_light, x_heavy = input_features.split([self.c_light, self.c_heavy], dim=1)

        # --- Process Light Path ---
        out_light = self.conv_shortcut(x_light)

        # --- Process Heavy Path ---
        features = [x_heavy]
        current_input_to_dense_layer = x_heavy

        for i in range(self.num_layers):
            if i > 0:
                concat_for_shuffle = torch.cat(features, dim=1)
                num_parts_for_shuffle = len(features)
                shuffled_features = channel_shuffle(concat_for_shuffle, groups=num_parts_for_shuffle)
                current_input_to_dense_layer = self.reduction_modules[i - 1](shuffled_features)

            layer_output = self.layers[i](current_input_to_dense_layer)
            features.append(layer_output)

        final_block_output = torch.cat(features, dim=1)
        shuffled_final_output = channel_shuffle(final_block_output, groups=len(features))
        out_heavy = self.final_reduction(shuffled_final_output)

        # --- Concatenate and Fuse ---
        out = torch.cat((out_light, out_heavy), dim=1)
        return self.final_conv(self.final_fusion(out))


class DenseRepViTBlock_EGA(nn.Module):
    def __init__(self, c1, c2, constant, num_layers,use_se,stride=1,use_hs=True):
        super().__init__()
        assert c1 == c2
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.constant = constant

        for _ in range(num_layers):
            self.layers.append (
                RepViTBlock_ECA(c1, c2, use_se=use_se, stride=stride, use_hs=use_hs
                )
            )
        self.reduction_modules = nn.ModuleList()
        for i in range(num_layers-1):
            num_tensors_to_concatenate = i + 2
            in_channels_for_reduction = num_tensors_to_concatenate * constant

            self.reduction_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_for_reduction, self.constant, kernel_size=1, bias=False),
                    nn.BatchNorm2d(constant),
                    nn.GELU() # 或者 nn.ReLU(inplace=True)
                )
            )
        final_in_channels = (num_layers + 1) * constant
        self.final_reduction = nn.Sequential(
            nn.Conv2d(final_in_channels, self.constant, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.constant),
            nn.GELU()
        )


    def forward(self, input_features: Tensor) -> Tensor:

        features = [input_features]
        current_input_to_dense_layer = input_features

        for i in range(self.num_layers):
            if i > 0:
               concat_for_shuffle = torch.cat(features, dim=1)
               num_parts_for_shuffle = len(features)
               shuffled_features = channel_shuffle(concat_for_shuffle, groups=num_parts_for_shuffle)

               current_input_to_dense_layer = self.reduction_modules[i-1](shuffled_features)

            
            layer_output = self.layers[i](current_input_to_dense_layer)
            features.append(layer_output)
        
        final_block_output = torch.cat(features, dim=1)
        shuffled_final_output = channel_shuffle(final_block_output, groups=len(features))
        final_output = self.final_reduction(shuffled_final_output)
        
        return final_output

class DenseRepViTBlock_(nn.Module):
    def __init__(self, c1, c2, constant, num_layers,use_se,stride=1,use_hs=True):
        super().__init__()
        assert c1 == c2
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.constant = constant

        for _ in range(num_layers):
            self.layers.append (
                RepViTBlock(c1, c2, use_se=use_se, stride=stride, use_hs=use_hs
                )
            )
        self.reduction_modules = nn.ModuleList()
        for i in range(num_layers-1):
            num_tensors_to_concatenate = i + 2
            in_channels_for_reduction = num_tensors_to_concatenate * constant

            self.reduction_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_for_reduction, self.constant, kernel_size=1, bias=False),
                    nn.BatchNorm2d(constant),
                    nn.GELU() # 或者 nn.ReLU(inplace=True)
                )
            )
        final_in_channels = (num_layers + 1) * constant
        self.final_reduction = nn.Sequential(
            nn.Conv2d(final_in_channels, self.constant, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.constant),
            nn.GELU()
        )


    def forward(self, input_features: Tensor) -> Tensor:

        features = [input_features]
        current_input_to_dense_layer = input_features

        for i in range(self.num_layers):
            if i > 0:
               concat_for_shuffle = torch.cat(features, dim=1)
               num_parts_for_shuffle = len(features)
               shuffled_features = channel_shuffle(concat_for_shuffle, groups=num_parts_for_shuffle)

               current_input_to_dense_layer = self.reduction_modules[i-1](shuffled_features)

            
            layer_output = self.layers[i](current_input_to_dense_layer)
            features.append(layer_output)
        
        final_block_output = torch.cat(features, dim=1)
        shuffled_final_output = channel_shuffle(final_block_output, groups=len(features))
        final_output = self.final_reduction(shuffled_final_output)
        
        return final_output


class ChannelAttention(nn.Module):
    """
    论文中描述的通道注意力模块 (Channel Attention Module, CA)。
    这个实现严格遵循了公式(5)的逻辑。
    """
    def __init__(self, in_channels, reduction_ratio=16):
        """
        初始化通道注意力模块。
        :param in_channels: 输入特征图的通道数。
        :param reduction_ratio: 通道缩减率r，用于MLP的瓶颈层。论文中没有明确给出，但16是常用值。
        """
        super(ChannelAttention, self).__init__()
        # 检查缩减率是否合理
        if in_channels <= reduction_ratio:
            # 如果输入通道数本身就很小，直接使用输入通道数的一半或者1作为中间通道
            # 避免降维后通道数为0或负数
            mip_channels = in_channels // 2 if in_channels > 1 else 1
        else:
            mip_channels = in_channels // reduction_ratio
        # 1. Squeeze 操作: 全局平均池化和全局最大池化
        # 这两个操作在forward函数中直接调用F.adaptive_avg_pool2d和F.adaptive_max_pool2d实现
        # 所以这里不需要定义层
        # 2. Shared MLP: 一个共享的多层感知机
        # 使用1x1卷积来实现全连接层，这是CNN中的标准做法
        self.shared_mlp = nn.Sequential(
            # 第一个1x1卷积，对应 W0，用于降维
            nn.Conv2d(in_channels, mip_channels, kernel_size=1, bias=False),
            # ReLU激活函数，对应 δ
            nn.ReLU(inplace=True),
            # 第二个1x1卷积，对应 W1，用于升维
            nn.Conv2d(mip_channels, in_channels, kernel_size=1, bias=False)
        )
        
        # 3. Sigmoid 激活函数，对应 σ
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播过程。
        :param x: 输入特征图，尺寸 (B, C, H, W)
        :return: 经过通道注意力加权后的特征图，尺寸 (B, C, H, W)
        """
        # 保存原始输入，用于最后的乘法
        original_input = x
        # 获取输入尺寸
        B, C, H, W = x.size()
        # Squeeze 操作
        # 全局平均池化 -> (B, C, 1, 1)
        avg_pool_out = F.adaptive_avg_pool2d(x, (1, 1))
        # 全局最大池化 -> (B, C, 1, 1)
        max_pool_out = F.adaptive_max_pool2d(x, (1, 1))
        # Shared MLP 操作
        # 分别通过共享的MLP
        avg_mlp_out = self.shared_mlp(avg_pool_out)
        max_mlp_out = self.shared_mlp(max_pool_out)
        # Merge 操作: 元素级相加
        merged_out = avg_mlp_out + max_mlp_out
        # Excitation 操作: Sigmoid
        attention_weights = self.sigmoid(merged_out)
        # Reweight 操作: 元素级相乘
        # 利用广播机制 (B, C, 1, 1) -> (B, C, H, W)
        output = original_input * attention_weights
        
        return output

class FeatureFusionAttention(nn.Module):
    """
    A module to fuse two input feature maps, process them through a 
    depthwise separable convolution, and apply channel attention.
    """
    def __init__(self, c1, c_out):
        """
        Initializes the feature fusion and attention module.
        :param c1: The combined number of input channels from the two sources (auto-calculated by parser).
        :param c_out: The desired number of output channels.
        """
        super(FeatureFusionAttention, self).__init__()
        self.dw_conv = DWConv(c1, c_out)
        self.channel_attention = ChannelAttention(c_out)

    def forward(self, inputs):
        """
        Forward pass for the fusion module.
        :param inputs: A list of two tensors [x1, x2]. 
                       x1 is assumed to have the target spatial dimensions (e.g., H/4, W/4).
        :return: The processed feature map.
        """
        x1, x2 = inputs
        
        # Resize the second input (x2) to match the spatial dimensions of the first (x1)
        if x2.shape[2:] != x1.shape[2:]:
            x2_resized = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        else:
            x2_resized = x2
            
        # Concatenate the features along the channel dimension
        fused_features = torch.cat([x1, x2_resized], dim=1)
        
        # Pass through the depthwise separable convolution
        extracted_features = self.dw_conv(fused_features)
        
        # Apply channel attention
        final_output = self.channel_attention(extracted_features)
        
        return final_output

class LoGFilter(nn.Module):
    """
    高斯-拉普拉斯滤波器模块。
    这个实现借鉴自legnet.py，但进行了简化和适配，使其自包含。
    """
    def __init__(self, in_c, out_c, kernel_size, sigma):
        super(LoGFilter, self).__init__()
        
        # --- 创建并固定LoG卷积核 ---
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        
        # 高斯-拉普拉斯公式
        term1 = (xx**2 + yy**2 - 2 * sigma**2) / (2 * math.pi * sigma**4)
        term2 = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = term1 * term2
        
        # 归一化
        kernel = kernel - kernel.mean()
        # if torch.sum(kernel) != 0:
        #     kernel = kernel / torch.sum(kernel)
        l1_norm = torch.sum(torch.abs(kernel))
        if l1_norm > 1e-6: # 增加一个小的epsilon来判断
            kernel = kernel / l1_norm
            
        log_kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(out_c, 1, 1, 1)

        # LoG滤波层
        self.log_conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=out_c, bias=False)
        self.log_conv.weight.data = log_kernel
        self.log_conv.weight.requires_grad = False

        # 后续处理层 (硬编码)
        self.norm = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()

    def forward(self, x):
        log_features = self.log_conv(x)
        log_edge = self.act(self.norm(log_features))
        return log_edge
    
class EG_stem(nn.Module):
    """
    A simplified stem block that applies a single depthwise convolution
    and adds the result back to the input (residual connection).
    """
    def __init__(self, channels):
        super(EG_stem, self).__init__()
        self.dw_conv = DWConv(channels, channels)

    def forward(self, x):
        return x + self.dw_conv(x)
    

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
    
class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)

from other_model.InternImage.detection.ops_dcnv3 import modules as dcnv3
# 假设以下模块已按需导入
import torch
import torch.nn as nn

# --- 辅助模块 (确保这些定义存在) ---

class to_channels_first(nn.Module):
    """将张量从 (B, H, W, C) 转换为 (B, C, H, W)"""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.permute(0, 3, 1, 2)

class to_channels_last(nn.Module):
    """将张量从 (B, C, H, W) 转换为 (B, H, W, C)"""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.permute(0, 2, 3, 1)

class DeformableDW(nn.Module):
    def __init__(self, c1, c2, stride, offset_scale):
        super().__init__()
        self.to_last = to_channels_last()
        self.to_first = to_channels_first()
        # DCN本身不改变通道数，所以它在 c1 通道上工作
        self.dcn = dcnv3.DCNv3(
            channels=c1,
            kernel_size=3,
            stride=stride,
            pad=1,
            group=c1, # 深度可分离
            offset_scale=offset_scale
        )
        self.bn_dcn = nn.BatchNorm2d(c1)
        
        # 使用一个 1x1 卷积来改变通道数从 c1 -> c2
        self.pw_conv = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(c2)

    def forward(self, x):
        # DCN (c1 -> c1)
        x_last = self.to_last(x)
        deformed_x_last = self.dcn(x_last)
        deformed_x = self.to_first(deformed_x_last)
        x_dcn_bn = self.bn_dcn(deformed_x)
        
        # Pointwise Conv (c1 -> c2)
        output = self.pw_conv(x_dcn_bn)
        return self.bn_pw(output)

class DeformableViTBlock(nn.Module):
    """
    一个集成了可变型深度卷积和 FFN 的完整块。
    此版本为适配 YOLOv8 的配置文件和解析器而特别修改。
    """
    def __init__(self, c1, c2, stride=1, mlp_ratio=2, use_se=True, offset_scale=1.0, act_layer=nn.GELU):
        """
        Args:
            c1 (int): 输入通道数.
            c2 (int): 输出通道数.
            stride (int): 步长, 1 或 2.
            mlp_ratio (float): FFN 隐藏层的扩展比例.
            offset_scale (float): DCNv3 的偏移缩放系数.
            act_layer (nn.Module): 激活函数.
        """
        super().__init__()
        # 确保 stride=1 时输入输出通道一致，这是 YOLO 中非下采样块的常见约束
        if stride == 1:
            assert c1 == c2, "stride=1时, c1 和 c2 必须相等！"
        
        hidden_channels = int(c2 * mlp_ratio)

        # 1. 可变型卷积部分 (Token Mixer)
        #    它的任务是将 c1 -> c2，同时根据 stride 进行下采样
        self.token_mixer = nn.Sequential(
            DeformableDW(c1, c2, stride, offset_scale),
            # SE模块在 DeformableDW 的输出 c2 上工作
            SqueezeExcite(c2) if use_se else nn.Identity() 
        )

        # 2. FFN 部分 (Channel Mixer)
        #    它在 c2 通道上工作，并带有残差连接
        self.norm = nn.BatchNorm2d(c2)
        self.ffn = nn.Sequential(
            nn.Conv2d(c2, hidden_channels, kernel_size=1, bias=False),
            act_layer(),
            nn.Conv2d(hidden_channels, c2, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # 1. Token Mixer (c1 -> c2)
        x = self.token_mixer(x)
        
        # 2. FFN with residual (在 c2 上工作)
        x = x + self.ffn(self.norm(x))
        
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)

class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.Gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.Gelu(x)

        return x

class Eage_detect(nn.Module):
    """
    边缘门控模块 (已重构为使用SobelGate子模块)。
    
    功能: 对输入的单个特征图，通过SobelGate模块获取边缘门控信号，
          与原始输入相乘，最后通过一个DW卷积处理。
    """
    def __init__(self, c1):
        """
        Args:
            c1 (int): 输入特征图的通道数。
        """
        super(Eage_detect, self).__init__()
        
        # 1. 实例化Sobel门控信号生成器
        self.sobel_gate = SobelGate(c1)
        
        # 2. 最后的深度卷积，输入输出通道数不变
        self.dw_conv_out = DWConv(c1, c1, kernel_size=3)

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入特征图，形状为 [B, C, H, W]。
        """
        # a. 使用SobelGate模块获取门控信号
        gate_signal = self.sobel_gate(x)
        
        # b. 与原始输入相乘
        gated_features = x * gate_signal
        
        # c. 经过一个DW卷积
        output = self.dw_conv_out(gated_features)
        
        return output

class Edge_guide(nn.Module):
    """
    (已修改)
    A module to fuse two input feature maps. It aligns the spatial dimensions of
    the first input (x1) to match the second (x2) using fixed 2x pooling.
    The pooling combines both max and average pooling for richer features.
    """
    def __init__(self, c_in1, c_in2, c_out):
        """
        Initializes the Edge_guide module.
        Args:
            c_in1 (int): Number of channels for the first input (x1).
            c_in2 (int): Number of channels for the second input (x2).
            c_out (int): Number of output channels.
        """
        super(Edge_guide, self).__init__()
        # 1. Reduction convolution for the resized x1.
        # Input will be concatenation of max-pooled x1 and avg-pooled x1,
        # so the channel count is 2 * c_in1. Output should match original x1.
        self.resize_reduction = Conv(c_in1 * 2, c_in1, 1)

        # 2. DWConv receives the concatenation of resized x1 (c_in1) and x2 (c_in2).
        self.dw_conv = DWConv(c_in1 + c_in2, c_out)
        self.channel_attention = ECA(c_out)
        self.sa = SAM()

    def forward(self, inputs):
        """
        Forward pass for the fusion module.
        :param inputs: A list of two tensors [x1, x2].
                       x2 is assumed to have the target spatial dimensions.
        """
        x1, x2 = inputs

        # --- 核心修改: 尺寸对齐逻辑 ---
        if x1.shape[2:] != x2.shape[2:]:
            # If downsampling is needed (assuming 2x)
            if x1.shape[2] > x2.shape[2]:
                # 1. Create two feature maps using fixed 2x pooling
                x1_max = F.max_pool2d(x1, kernel_size=2, stride=2)
                x1_avg = F.avg_pool2d(x1, kernel_size=2, stride=2)
                
                # 2. Concatenate them along the channel dimension
                x1_cat = torch.cat([x1_max, x1_avg], dim=1)
                
                # 3. Use a 1x1 convolution to reduce dimensions and fuse information
                x1_resized = self.resize_reduction(x1_cat)

            # If upsampling is needed (fallback case)
            else:
                x1_resized = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        else:
            # If shapes are already the same, no resizing is needed
            x1_resized = x1

        # --- 后续功能 (保持不变) ---
        fused_features = torch.cat([x1_resized, x2], dim=1)
        extracted_features = self.dw_conv(fused_features)
        final_output = self.channel_attention(extracted_features)
        final_output = final_output*self.sa(final_output)

        return final_output

# class Edge_guide(nn.Module):
#     """
#     (已修改)
#     A module to fuse two input feature maps, ensuring the output spatial 
#     dimensions match the SECOND input (x2), which is assumed to be P2 (H/4, W/4).
#     """
#     def __init__(self, c1, c_out):
#         super(Edge_guide, self).__init__()
#         self.dw_conv = DWConv(c1, c_out)
#         self.channel_attention = ChannelAttention(c_out)

#     def forward(self, inputs):
#         """
#         Forward pass for the fusion module.
#         :param inputs: A list of two tensors [x1, x2].
#                        x2 is now assumed to have the target spatial dimensions (H/4, W/4).
#         """
#         x1, x2 = inputs
        
#         # --- 核心修改 ---
#         # 以 x2 的尺寸为基准，对齐 x1
#         if x1.shape[2:] != x2.shape[2:]:
#             # x1 (来自Eage_detect, P1尺寸) 通常比 x2 (来自主干, P2尺寸) 大，所以这里是下采样
#             if x1.shape[2] > x2.shape[2]:
#                 if x1.shape[2] % x2.shape[2] == 0:
#                     stride = x1.shape[2] // x2.shape[2]
#                     x1_resized = F.max_pool2d(x1, kernel_size=stride, stride=stride)
#                 else:
#                     x1_resized = F.adaptive_max_pool2d(x1, x2.shape[2:])
#             # 如果x1更小，则上采样 (以防万一)
#             else:
#                 x1_resized = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
#         else:
#             x1_resized = x1
            
#         # Concatenate a resized x1 with the original x2
#         # 注意拼接顺序可以调整，这里保持x2在后，与输入顺序一致
#         fused_features = torch.cat([x1_resized, x2], dim=1)
        
#         extracted_features = self.dw_conv(fused_features)
#         final_output = self.channel_attention(extracted_features)
        
#         return final_output


class Fuse_Features(nn.Module):
    """
    (已重构)
    A robust module to fuse two input feature maps (x1, x2).
    It aligns the spatial dimensions of x2 to match x1. If downsampling is needed,
    it uses a combination of max and average pooling for richer feature preservation.
    Finally, it processes the fused tensor with a reduction convolution and attention.
    """
    def __init__(self, c_in1, c_in2, c_out):
        """
        Initializes the Fuse_Features module.
        :param c_in1: Number of channels for the first input feature map (x1).
        :param c_in2: Number of channels for the second input feature map (x2).
        :param c_out: The desired number of output channels.
        """
        super().__init__()
        # Convolution to process the multi-pooled (max + avg) x2 during resizing.
        # Input is 2 * c_in2, output is c_in2.
        self.resize_reduction = Conv(c_in2 * 2, c_in2, 1)

        # Main reduction convolution for the final fused features.
        # Input is concatenation of x1 (c_in1) and aligned x2 (c_in2).
        self.reduction = Conv(c_in1 + c_in2, c_out, 1)
        
        # Standard CBAM-style attention: ECA (channel) then SAM (spatial)
        self.eca = ECA(c_out)
        self.sa = SAM()

    def _align_features(self, x1, x2):
        """
        Aligns the spatial dimensions of x2 to match x1.
        """
        h1, w1 = x1.shape[2:]
        h2, w2 = x2.shape[2:]

        if h1 == h2 and w1 == w2:
            return x2

        # Downsample x2 if it's larger than x1, using the multi-pool strategy
        if h2 > h1 or w2 > w1:
            # Use fixed pooling if it's a simple integer downscale, otherwise adaptive
            if h2 % h1 == 0 and w2 % w1 == 0:
                stride_h, stride_w = h2 // h1, w2 // w1
                x2_max = F.max_pool2d(x2, kernel_size=(stride_h, stride_w), stride=(stride_h, stride_w))
                x2_avg = F.avg_pool2d(x2, kernel_size=(stride_h, stride_w), stride=(stride_h, stride_w))
            else: # Fallback to adaptive pooling for non-integer scales
                x2_max = F.adaptive_max_pool2d(x2, (h1, w1))
                x2_avg = F.adaptive_avg_pool2d(x2, (h1, w1))
            
            # Concatenate pooled features and reduce
            x2_cat = torch.cat([x2_max, x2_avg], dim=1)
            return self.resize_reduction(x2_cat)

        # Upsample x2 if it's smaller than x1
        else:
            return F.interpolate(x2, size=(h1, w1), mode='bilinear', align_corners=False)

    def forward(self, x):
        """
        Forward pass for the fusion module.
        """
        x1, x2 = x
        
        # 1. Align the spatial dimensions of x2 to match x1 FIRST.
        # This is done on the original x2 and incorporates the multi-pool logic.
        x2_aligned = self._align_features(x1, x2)
            
        # 2. Concatenate the original x1 and the aligned x2.
        fused = self.reduction(torch.cat([x1, x2_aligned], 1))
        
        # 3. Apply attention mechanisms sequentially (CBAM style).
        fused_after_eca = self.eca(fused)
        attention_map_sa = self.sa(fused_after_eca)
        final_output = fused_after_eca * attention_map_sa

        return final_output


class Edge_Emphasize(nn.Module):
    def __init__(self, channel):
        super(Edge_Emphasize, self).__init__()
        t = int(abs((math.log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = nn.Sequential(
            # 深度卷积 (Depthwise)
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.BatchNorm2d(channel),
            nn.GELU(),
            
            # 逐点卷积 (Pointwise)
            nn.Conv2d(channel, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.GELU()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.SAM = SAM()
        self.final_conv = ConvBNR(channel, channel)

    def forward(self, inputs):
        # The model parser provides inputs as a list for multi-source layers.
        # We unpack it here to match the user's desired `(c, att)` signature.
        in1, in2 = inputs
        # Intelligently assign feature map 'c' and attention 'att' based on channel count
        if in1.shape[1] == 1 and in2.shape[1] > 1:
            att, c = in1, in2
        else: # Default to assuming [feature, attention] order, which is the corrected YAML order
            c, att = in1, in2

        # This is the user-specified forward logic.
        # The spatial dimension check is more robust than comparing the full tensor size tuples.
        if c.shape[2:] != att.shape[2:]:
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        
        x = c * att + c
        x = self.conv2d(x)
        wei = self.avg_pool(x)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = x * wei
        SAM_wei = self.SAM(x)
        out = x * SAM_wei

        out = out+c
        out = self.final_conv(out)

        return out


import torch
import torch.nn as nn

class DWConv(nn.Module):
    """
    一个已经定义好的深度可分离卷积模块 (假设)。
    注意：这里我假设 DWConv 接收 in_channels 和 out_channels。
    如果您的 DWConv 实现不同 (例如，只接收一个channel参数)，您可能需要稍作调整。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act=nn.GELU):
        super(DWConv, self).__init__()
        # 自动计算padding以保持空间尺寸
        padding = (kernel_size - 1) // 2
        
        self.sequential_ops = nn.Sequential(
            # 深度卷积
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            act(),
            
            # 逐点卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act()
        )

    def forward(self, x):
        return self.sequential_ops(x)

from torch.cuda.amp import autocast
class Scharr(nn.Module):
    def __init__(self, channel):
        super(Scharr, self).__init__()
        self.epsilon = 1e-6
        # 定义Scharr滤波器
        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # 使用固定的、不可学习的卷积层来实现Scharr滤波
        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        
        self.conv_x.weight.data = scharr_x.repeat(channel, 1, 1, 1)
        self.conv_y.weight.data = scharr_y.repeat(channel, 1, 1, 1)
        
        # 权重不可学习
        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False

        # 使用标准的PyTorch层
        self.norm = nn.BatchNorm2d(channel)
        self.act = nn.GELU() # 或者 nn.ReLU()
        self.conv_extra = DWConv(channel, channel) # 使用自包含的DWConv

        # self.register_forward_hook(inspect_forward_output)
        # print(f"✅ Forward hook registered on instance of {self.__class__.__name__} with {channel} channels.")

    def forward(self, x):
        # ==========================================================
        # forward 方法是唯一被修改的部分
        # 核心思想：显式地将计算过程转换到 float32，完成后再转换回原始类型。
        # ==========================================================
        # --- 核心改动 1: 保存输入的原始数据类型 ---
        # 这样模块在计算完成后可以恢复它，对外部网络透明。
        input_dtype = x.dtype
        # --- 核心改动 2: 将输入显式转换为 float32 ---
        # 这是确保后续所有计算都在高精度下进行的最可靠方法。
        x_f32 = x.to(torch.float32)
        # (移除了原有的 with autocast(...) 块，因为它不起作用)
        # 1. Scharr 卷积 (现在输入是 x_f32，所以计算是 float32)
        edges_x = self.conv_x(x_f32)
        edges_y = self.conv_y(x_f32)
    
        # 2. 计算平方和 (在 float32 上进行，安全无溢出风险)
        gradient_magnitude_squared = edges_x.pow(2) + edges_y.pow(2)
    
        # 3. 开方 (在 float32 上进行，结果更精确)
        gradient_magnitude = torch.sqrt(gradient_magnitude_squared + self.epsilon)
    
        # 4. 后续处理 (仍在 float32 下进行)
        scharr_edge = self.act(self.norm(gradient_magnitude))
        
        # 5. 残差连接 (两个操作数 x_f32 和 scharr_edge 都是 float32)
        fused_output = self.conv_extra(x_f32 + scharr_edge)
        
        # --- 核心改动 3: 将最终输出转换回原始输入类型 ---
        # 这使得模块成为一个“良好公民”，无缝衔接外部的AMP上下文。
        # 如果外部是float16，它就输出float16；如果外部是float32，它就输出float32。
        return fused_output.to(input_dtype)

class Sobel(nn.Module):
    def __init__(self, channel):
        super(Sobel, self).__init__()
        self.epsilon = 1e-6
        
        # 定义Sobel滤波器
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # 使用固定的、不可学习的卷积层来实现Sobel滤波
        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        
        self.conv_x.weight.data = sobel_x.repeat(channel, 1, 1, 1)
        self.conv_y.weight.data = sobel_y.repeat(channel, 1, 1, 1)
        
        # 权重不可学习
        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False

        # 使用标准的PyTorch层
        self.norm = nn.BatchNorm2d(channel)
        self.act = nn.GELU() # 或者 nn.ReLU()
        self.conv_extra = DWConv(channel, channel)

    def forward(self, x):
        # forward 方法与您的 Scharr 模块完全相同，以确保混合精度下的稳定性
        
        # 1. 保存输入的原始数据类型
        input_dtype = x.dtype
        
        # 2. 将输入显式转换为 float32
        x_f32 = x.to(torch.float32)
        
        # 3. Sobel 卷积 (在 float32 上进行)
        edges_x = self.conv_x(x_f32)
        edges_y = self.conv_y(x_f32)
    
        # 4. 计算梯度幅值的平方 (在 float32 上进行)
        gradient_magnitude_squared = edges_x.pow(2) + edges_y.pow(2)
    
        # 5. 开方 (在 float32 上进行)
        gradient_magnitude = torch.sqrt(gradient_magnitude_squared + self.epsilon)
    
        # 6. 后续处理 (仍在 float32 下进行)
        sobel_edge = self.act(self.norm(gradient_magnitude))
        
        # 7. 残差连接 (两个操作数 x_f32 和 sobel_edge 都是 float32)
        fused_output = self.conv_extra(x_f32 + sobel_edge)
        
        # 8. 将最终输出转换回原始输入类型
        return fused_output.to(input_dtype)


class Conv_Extra(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   build_norm_layer(norm_layer, channel)[1])
    def forward(self, x):
        out = self.block(x)
        return out

# class Scharr(nn.Module):
#     # __init__ 方法保持和上次一样，我们只修改 forward
#     def __init__(self, dim):
#         super(Scharr, self).__init__()
#         # ... __init__ 的所有内容保持不变 ...
#         scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], dtype=torch.float32).view(1, 1, 3, 3)
#         scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], dtype=torch.float32).view(1, 1, 3, 3)
#         self.conv_x = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
#         self.conv_y = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
#         self.conv_x.weight.data.copy_(scharr_x.repeat(dim, 1, 1, 1))
#         self.conv_y.weight.data.copy_(scharr_y.repeat(dim, 1, 1, 1))
#         self.conv_x.weight.requires_grad = False
#         self.conv_y.weight.requires_grad = False
        
#         # The try-except block has been removed to eliminate the mmcv dependency.
#         # We now directly use the PyTorch-native layers from the original 'except' block.
#         # self.norm = nn.BatchNorm2d(dim)
#         self.act = nn.GELU()
#         self.conv_extra = DWConv(dim, dim) 
        
#         self.epsilon = 1e-6
#         # 暂时移除Hook，因为我们将手动进行更详细的调试
#         # self.register_forward_hook(inspect_forward_output)
#     def forward(self, x):
#         # 你的 Scharr.forward 代码应该看起来像这样，包含了所有的调试点
        
#         # 使用 autocast 来确保在正确的精度下运行
#         with autocast(enabled=True): 
#             # 确保计算在 float32 下进行，以获得最大精度和范围
#             x_f32 = x.to(torch.float32)
#             debug_tensor("Input 'x' (converted to float32)", x_f32) # 调试点0
#             edges_x = self.conv_x(x_f32)
#             debug_tensor("Step 1: 'edges_x' after conv_x", edges_x) # 调试点1
#             edges_y = self.conv_y(x_f32)
#             debug_tensor("Step 2: 'edges_y' after conv_y", edges_y) # 调试点2
            
#             gradient_magnitude_squared = edges_x.float()**2 + edges_y.float()**2 + 1e-6
#             debug_tensor("Step 3: Gradient Magnitude Squared", gradient_magnitude_squared) # 调试点3
#             # 这里我们先用回最原始的 sqrt，因为我们想找到问题的根源
#             # 你可以先注释掉 clamp 版本，用回这个来复现问题
#             # gradient_magnitude = torch.sqrt(gradient_magnitude_squared + self.epsilon) 
#             gradient_magnitude = torch.sqrt(gradient_magnitude_squared.clamp(min=1e-6)) # 或者就用clamp版本
#             debug_tensor("Step 4: Gradient Magnitude after sqrt", gradient_magnitude) # 调试点4
#             # norm_output = self.norm(gradient_magnitude)
#             # debug_tensor("Step 5: Output of norm_layer", norm_output) # 调试点5
            
#             act_output = self.act(gradient_magnitude)
#             debug_tensor("Step 5: Output of activation", act_output) # 调试点6
            
#             out_f32 = self.conv_extra(x_f32 + act_output)
#             debug_tensor("Step 6: Final Output (float32)", out_f32) # 调试点7
#         # 检查最终输出是否有NaN
#         if torch.isnan(out_f32).any():
#             # 我们现在可以确信，如果触发了这个，上面的日志一定已经被打印出来了
#             raise RuntimeError("NaN DETECTED! Stopping training for inspection. Check the logs above.")
#         # 将输出转换回输入的原始类型
#         return out_f32.to(x.dtype)

class EGA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 分支 A: 边缘分支
        self.edge_sobel_x = EdgeConvSobelX(dim, dim)
        self.edge_sobel_y = EdgeConvSobelY(dim, dim)
        self.conv_reduce = Conv(dim * 2, dim, k=1, act=nn.GELU())
        self.sa = SAM()

        # 分支 B: 标准 3x3 卷积分支
        self.standard_conv_branch = Conv(dim, dim, k=3, s=1, p=None, act=True)

        # 融合两个分支的 3x3 卷积
        self.fusion_conv = Conv(dim * 2, dim, k=3, s=1, p=None, act=True)

        # 后续的 ECA 注意力模块
        self.eca = ECA(dim)

    def forward(self, x):
        # 1. 计算边缘分支 (SAM 已从此移除)
        edge_x = self.edge_sobel_x(x)
        edge_y = self.edge_sobel_y(x)
        edge_cat = torch.cat([edge_x, edge_y], dim=1)
        edge_features = self.conv_reduce(edge_cat)

        # 2. 计算标准卷积分支
        conv_features = self.standard_conv_branch(x)

        # 3. 拼接并用 3x3 卷积融合
        concatenated_features = torch.cat([conv_features, edge_features], dim=1)
        fused = self.fusion_conv(concatenated_features)

        # 4. 先进行残差连接
        residual_out = fused + x

        # 5. 再应用 ECA 和 SA 注意力
        eca_output = self.eca(residual_out)
        sa_attention = self.sa(eca_output)
        final_output = eca_output * sa_attention

        return final_output


# class EGA(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
        
#         # --- 1. 定义并行的两个分支 ---
#         # 分支 A: 边缘分支 (无变化)
#         self.edge_sobel_x = EdgeConvSobelX(dim, dim)
#         self.edge_sobel_y = EdgeConvSobelY(dim, dim)
#         # 这里的 Conv2d_BN 应该也是您项目中的一个标准模块，这里假设它存在
#         self.conv_reduce = Conv2d_BN(dim * 2, dim, ks=1) 
#         self.sa = SAM()
        
#         # --- 分支 B: 标准卷积分支 (已修改) ---
#         # 使用您提供的标准化 Conv 模块
#         # k=3 (3x3卷积), s=1, p=None (让 autopad 自动计算 padding)
#         # act=True 使用默认的 SiLU 激活函数
#         self.standard_conv_branch = ResidualBlock(dim, dim, act=nn.GELU())
        
#         # --- 2. 定义融合模块 (无变化) ---
#         self.fusion = MAFM_Fusion(dim)
        
#         # --- 3. 定义后续的注意力模块 (无变化) ---
#         self.eca = ECA(dim)

#     def forward(self, x):
#         # 1. 并行计算两个分支的输出
#         edge_x = self.edge_sobel_x(x)
#         edge_y = self.edge_sobel_y(x)
#         edge_cat = torch.cat([edge_x, edge_y], dim=1)
#         edge_reduced = self.conv_reduce(edge_cat)
#         edge_features = self.sa(edge_reduced) * edge_reduced
        
#         conv_features = self.standard_conv_branch(x)
        
#         # 2. 使用 MAFM 动态融合两个分支
#         fused = self.fusion(conv_features, edge_features)
        
#         # 3. 通过后续的注意力模块进行特征精炼
#         eca_att = self.eca(fused)
#         # sam_attention = self.sa(eca_att)
#         # temp_output = eca_att * sam_attention
        
#         # 4. 最终的残差连接
#         final_output = eca_att + x

#         return final_output

class EGA_Conv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        # --- 核心改动 ---
        # 原来的 self.sobel 被替换为一个 3x3 的 DWConv
        # 我们称之为 feature_extractor 以表明其通用性
        self.feature_extractor = DWConv(dim, dim)
        
        # 后续的 dwconv 保持不变，可以称之为 fusion_conv
        self.fusion_conv = DWConv(dim, dim)
        
        # 注意力模块保持不变
        self.sam = SAM()
        self.eca = ECA(channel=dim)

    def forward(self, x):
        # 1. 使用 DWConv 提取特征 (替代了 Sobel)
        extracted_features = self.feature_extractor(x)

        # 2. 特征交互 (逻辑与原版相同)
        # 将原始输入 x 与提取出的特征进行交互
        att = x * extracted_features + x
        
        # 3. 信息融合 (逻辑与原版相同)
        fused = self.fusion_conv(att)

        # 4. 后续注意力与残差连接 (逻辑与原版相同)
        eca_att = self.eca(fused)
        sam_attention = self.sam(eca_att)
        temp_output = eca_att * sam_attention
        final_output = temp_output + x

        return final_output

class EGA_singel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # __init__ 方法保持不变
        self.sobel = Sobel(dim)
        self.dwconv = DWConv(dim, dim)
        self.sam = SAM()
        self.eca = ECA(dim)

    def forward(self, x):
        edge_features = self.sobel(x)

        att = x * edge_features + x
        fused = self.dwconv(att)

        output = fused

        eca_att = self.eca(output)
        sam_attention = self.sam(eca_att)
        temp_output = eca_att * sam_attention
        final_output = temp_output + x

        return final_output


class EGA_att(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scharr = Scharr(dim)
        self.dwconv = DWConv(dim, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        edge_features = self.scharr(x)
        attention = self.sigmoid(edge_features)
        out = attention * x
        out = self.dwconv(out)
        return out


import torch
import torch.nn as nn
 
 
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
 
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
 
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
 
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MLCA(nn.Module):
    def __init__(self, in_size,local_size=5,gamma = 2, b = 1,local_weight=0.5):
        super(MLCA, self).__init__()

        # ECA 计算方法
        self.local_size=local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)   # eca  gamma=2
        k = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.local_weight=local_weight

        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool=nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        local_arv=self.local_arv_pool(x)
        global_arv=self.global_arv_pool(local_arv)

        b,c,m,n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape

        # (b,c,local_size,local_size) -> (b,c,local_size*local_size)-> (b,local_size*local_size,c)-> (b,1,local_size*local_size*c)
        temp_local= local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)

        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)


        # (b,c,local_size,local_size) <- (b,c,local_size*local_size)<-(b,local_size*local_size,c) <- (b,1,local_size*local_size*c)
        y_local_transpose=y_local.reshape(b, self.local_size * self.local_size,c).transpose(-1,-2).view(b,c, self.local_size , self.local_size)
        # y_global_transpose = y_global.view(b, -1).transpose(-1, -2).unsqueeze(-1)
        y_global_transpose = y_global.view(b, -1).unsqueeze(-1).unsqueeze(-1)  # 代码修正
        # print(y_global_transpose.size())
        # 反池化
        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(),[self.local_size, self.local_size])
        # print(att_local.size())
        # print(att_global.size())
        att_all = F.adaptive_avg_pool2d(att_global*(1-self.local_weight)+(att_local*self.local_weight), [m, n])
        # print(att_all.size())
        x=x*att_all
        return x
    
# class Context_Exploration_Block(nn.Module):
#     def __init__(self, input_channels):
#         super(Context_Exploration_Block, self).__init__()
#         self.input_channels = input_channels
#         self.channels_single = int(input_channels / 4)

#         self.p1_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
#         self.p2_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
#         self.p3_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
#         self.p4_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())

#         self.p1 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
#             nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
#         self.p1_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
#             nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())

#         self.p2 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
#             nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
#         self.p2_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
#             nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())

#         self.p3 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
#             nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
#         self.p3_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
#             nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())

#         self.p4 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
#             nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
#         self.p4_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
#             nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())

#         self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 3, 1, 1),
#                                     nn.GroupNorm(num_groups=32, num_channels=self.input_channels), nn.ReLU())

#     def forward(self, x):
#         p1_input = self.p1_channel_reduction(x)
#         p1 = self.p1(p1_input)
#         p1_dc = self.p1_dc(p1)

#         p2_input = self.p2_channel_reduction(x) + p1_dc
#         p2 = self.p2(p2_input)
#         p2_dc = self.p2_dc(p2)

#         p3_input = self.p3_channel_reduction(x) + p2_dc
#         p3 = self.p3(p3_input)
#         p3_dc = self.p3_dc(p3)

#         p4_input = self.p4_channel_reduction(x) + p3_dc
#         p4 = self.p4(p4_input)
#         p4_dc = self.p4_dc(p4)

#         ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

#         return ce
    
class Features_enhanced(nn.Module):
    """
    特征增强模块。
    融合主干特征和边缘图，并通过内嵌的深度可分离卷积和上下文探索模块进行增强。
    """
    def __init__(self, input_channels, edge_channels=256):
        """
        Args:
            input_channels (int): 主干输入特征 x 的通道数。
            edge_channels (int): 边缘模块输出的通道数 (默认为256)。
        """
        super().__init__()
        
        # 定义对齐层：如果边缘图的通道数与输入特征不匹配，则使用1x1卷积进行调整
        self.align_conv = nn.Identity()
        if edge_channels != input_channels:
            self.align_conv = nn.Conv2d(edge_channels, input_channels, kernel_size=1)
        
        # 核心处理层：直接在此定义深度可分离卷积
        self.dws_conv = nn.Sequential(
            # 1. 深度卷积 (Depthwise Conv)
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, groups=input_channels, bias=False),
            nn.GroupNorm(num_groups=min(32, input_channels), num_channels=input_channels),
            nn.ReLU(),
            # 2. 逐点卷积 (Pointwise Conv)
            nn.Conv2d(input_channels, input_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(32, input_channels), num_channels=input_channels),
            nn.ReLU(),
        )
        
        # 最后一步：上下文探索模块
        self.ce_block = Context_Exploration_Block(input_channels)

    def forward(self, x, edge_output):
        """
        Args:
            x: 主干网络输入特征, e.g., shape: [B, C, H, W]
            edge_output: Eage_detect模块的输出, e.g., shape: [B, C_edge, H_edge, W_edge]

        Returns:
            增强后的特征图, shape: [B, C, H, W]
        """
        # --- 步骤 1: 对齐边缘图并与输入相加 ---
        
        # a. 对齐通道数
        edge_aligned = self.align_conv(edge_output)
        
        # b. 对齐空间尺寸 (H, W)，以输入x为基准
        if edge_aligned.shape[2:] != x.shape[2:]:
            edge_aligned = F.interpolate(edge_aligned, size=x.shape[2:], mode='bilinear', align_corners=False)
            
        # c. 相加融合
        fused_features = x + edge_aligned
        
        # --- 步骤 2: 通过深度可分离卷积 ---
        dws_out = self.dws_conv(fused_features)
        
        # --- 步骤 3: 加入残差连接 (将输入x与dws_out相加) ---
        residual_out = x + dws_out
        
        # --- 步骤 4: 通过上下文探索模块进行最终增强 ---
        final_output = self.ce_block(residual_out)
        
        return final_output

class Features_enhance(nn.Module):
    """
    特征增强模块 (最新版)。
    通过对齐通道 -> concat -> channel_shuffle -> 降维的方式进行早期特征融合，
    再通过深度可分离卷积和上下文探索模块进行增强。
    """
    def __init__(self, input_channels, edge_channels=256):
        """
        Args:
            input_channels (int): 主干输入特征 x 的通道数。
            edge_channels (int): 边缘模块输出的通道数 (默认为256)。
        """
        super().__init__()
        
        # 步骤1.a: 定义边缘特征的通道对齐层
        # 这是为了确保concat的两个部分通道数相等，从而让channel_shuffle有效工作
        self.edge_align_conv = nn.Identity()
        if edge_channels != input_channels:
            self.edge_align_conv = nn.Conv2d(edge_channels, input_channels, kernel_size=1)
        
        # 步骤1.d: 定义降维层
        # 将concat和shuffle后的通道 (2 * input_channels) 降维回 input_channels
        self.channel_adjust_conv = nn.Conv2d(2 * input_channels, input_channels, kernel_size=1)

        # 步骤2: 核心处理层 - 深度可分离卷积
        self.dws_conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, groups=input_channels, bias=False),
            nn.GroupNorm(num_groups=min(32, input_channels), num_channels=input_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, input_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(32, input_channels), num_channels=input_channels),
            nn.ReLU(),
        )
        
        # 步骤4: 最后一步 - 上下文探索模块
        self.ce_block = Context_Exploration_Block(input_channels)

    def forward(self, x, edge_output):
        """
        Args:
            x (Tensor): 主干网络输入特征, e.g., shape: [B, C_in, H, W]
            edge_output (Tensor): Eage_detect模块的输出, e.g., shape: [B, C_edge, H_edge, W_edge]

        Returns:
            Tensor: 增强后的特征图, shape: [B, C_in, H, W]
        """
        # --- 步骤 1: 早期融合 (对齐 -> Concat -> Shuffle -> 降维) ---
        
        # a. 对齐边缘图的通道数
        edge_temp = self.edge_align_conv(edge_output)
        
        # b. 对齐边缘图的空间尺寸 (H, W)，以输入x为基准
        if edge_temp.shape[2:] != x.shape[2:]:
            edge_aligned = F.interpolate(edge_temp, size=x.shape[2:], mode='bilinear', align_corners=False)
        else:
            edge_aligned = edge_temp
            
        # c. 拼接两个通道数相同的特征
        concatenated_features = torch.cat([x, edge_aligned], dim=1)
        
        # d. 通道混洗 (groups=2 因为我们拼接了两个特征源)
        shuffled_features = channel_shuffle(concatenated_features, groups=2)
        
        # e. 使用1x1卷积降维，得到初步融合的特征
        fused_features = self.channel_adjust_conv(shuffled_features)
        
        # --- 步骤 2: 通过深度可分离卷积提取特征 ---
        dws_out = self.dws_conv(fused_features)
        
        # --- 步骤 3: 加入残差连接 (将原始输入x与dws_out相加) ---
        residual_out = x + dws_out
        
        # --- 步骤 4: 通过上下文探索模块进行最终增强 ---
        final_output = self.ce_block(residual_out)
        
        return final_output

# class Context_Exploration_Block(nn.Module):
#     """
#     集成了 C2f 结构、GELU 激活函数和 ECA 注意力机制的上下文探索模块。
#     """
#     # 更改 1: 将 e 的默认值设为 0.6
#     def __init__(self, c1, c2, e=0.6):
#         super(Context_Exploration_Block, self).__init__()
#         assert c1 == c2, "C1 and C2 should be equal for Context_Exploration_Block"
        
#         self.c_ = int(c2 * e)  # 用于处理和直连路径的通道数

#         self.cv1 = Conv(c1, 2 * self.c_, k=1, s=1)

#         self.channels_single = self.c_ // 4
        
#         def get_safe_gn_groups(num_channels, prefer=32):
#             num_channels = int(num_channels)
#             if num_channels == 0:
#                 return 1
#             for divisor in [prefer, 16, 8, 4, 2, 1]:
#                 if num_channels % divisor == 0:
#                     return divisor
#             return 1

#         gn_groups_single = get_safe_gn_groups(self.channels_single)
#         gn_groups_c = get_safe_gn_groups(self.c_)

#         # 更改 2: 将所有 nn.ReLU() 替换为 nn.GELU()
#         act = nn.GELU
#         self.p1_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.c_, self.channels_single, 1, 1, 0),
#             nn.GroupNorm(gn_groups_single, self.channels_single), act())
#         self.p2_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.c_, self.channels_single, 1, 1, 0),
#             nn.GroupNorm(gn_groups_single, self.channels_single), act())
#         self.p3_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.c_, self.channels_single, 1, 1, 0),
#             nn.GroupNorm(gn_groups_single, self.channels_single), act())
#         self.p4_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.c_, self.channels_single, 1, 1, 0),
#             nn.GroupNorm(gn_groups_single, self.channels_single), act())

#         self.p1 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
#             nn.GroupNorm(gn_groups_single, self.channels_single), act())
#         self.p1_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1, dilation=1),
#             nn.GroupNorm(gn_groups_single, self.channels_single), act())

#         self.p2 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
#             nn.GroupNorm(gn_groups_single, self.channels_single), act())
#         self.p2_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 2, dilation=2),
#             nn.GroupNorm(gn_groups_single, self.channels_single), act())

#         self.p3 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
#             nn.GroupNorm(gn_groups_single, self.channels_single), act())
#         self.p3_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 4, dilation=4),
#             nn.GroupNorm(gn_groups_single, self.channels_single), act())

#         self.p4 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
#             nn.GroupNorm(gn_groups_single, self.channels_single), act())
#         self.p4_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 8, dilation=8),
#             nn.GroupNorm(gn_groups_single, self.channels_single), act())

#         fusion_in_channels = 4 * self.channels_single
#         self.fusion = nn.Sequential(nn.Conv2d(fusion_in_channels, self.c_, 3, 1, 1),
#                             nn.GroupNorm(gn_groups_c, self.c_), act())
        
#         self.cv2 = Conv(2 * self.c_, c2, k=1, s=1)
        
#         # 更改 3: 在模块末尾初始化 ECA 层
#         self.eca = ECA(c2)

#     def forward(self, x):
#         y = list(self.cv1(x).chunk(2, 1))

#         y_process = y[1]
#         p1_input = self.p1_channel_reduction(y_process)
#         p1 = self.p1(p1_input)
#         p1_dc = self.p1_dc(p1)

#         p2_input = self.p2_channel_reduction(y_process) + p1_dc
#         p2 = self.p2(p2_input)
#         p2_dc = self.p2_dc(p2)

#         p3_input = self.p3_channel_reduction(y_process) + p2_dc
#         p3 = self.p3(p3_input)
#         p3_dc = self.p3_dc(p3)

#         p4_input = self.p4_channel_reduction(y_process) + p3_dc
#         p4 = self.p4(p4_input)
#         p4_dc = self.p4_dc(p4) 
        
#         ce_out = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

#         out = self.cv2(torch.cat((y[0], ce_out), 1))
        
#         # 更改 3: 在最终输出前应用 ECA
#         return self.eca(out)

class ScharrAttention(nn.Module):
    """
    一个使用Scharr算子生成空间注意力的模块。
    它可以作为一种轻量级的、无参数的空间注意力机制。
    """
    def __init__(self, use_sigmoid=True):
        """
        初始化Scharr注意力模块。
        :param use_sigmoid: bool, 如果为True，使用Sigmoid将边缘图转化为[0,1]的软性注意力权重 (流派二)。
                                如果为False，直接使用原始边缘图作为门控信号 (流派一)。
        """
        super(ScharrAttention, self).__init__()
        self.use_sigmoid = use_sigmoid
        # 定义Scharr卷积核 (不可训练)
        # Gx
        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]])
        # Gy
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]])
        # 将核变形为 (out_channels, in_channels/groups, H, W) 的格式
        # 这里 out_channels=1, in_channels/groups=1
        self.kernel_x = scharr_x.float().unsqueeze(0).unsqueeze(0)
        self.kernel_y = scharr_y.float().unsqueeze(0).unsqueeze(0)
        # 注册为 buffer，这样它会随模型移动到CPU/GPU，但不会被视为模型参数
        self.register_buffer('scharr_kernel_x', self.kernel_x)
        self.register_buffer('scharr_kernel_y', self.kernel_y)
        if self.use_sigmoid:
            self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        """
        前向传播。
        :param x: 输入特征图，尺寸为 (B, C, H, W)
        :return: 经过Scharr注意力加权后的特征图，尺寸不变。
        """
        B, C, H, W = x.size()
        # 为了对每个通道独立应用Scharr算子，我们使用分组卷积 (grouped convolution)
        # 将输入通道 C 作为分组数，这样每个卷积核只作用于一个输入通道
        # 扩展卷积核以匹配输入通道数
        kernel_x = self.scharr_kernel_x.repeat(C, 1, 1, 1)
        kernel_y = self.scharr_kernel_y.repeat(C, 1, 1, 1)
        # 使用 F.conv2d 进行卷积操作
        # padding=1 保持尺寸不变
        grad_x = F.conv2d(x, kernel_x, bias=None, stride=1, padding=1, groups=C)
        grad_y = F.conv2d(x, kernel_y, bias=None, stride=1, padding=1, groups=C)
        # 计算梯度幅度 (边缘强度图)
        # 添加 epsilon 防止 sqrt(0) 的NaN梯度
        edge_map = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        if self.use_sigmoid:
            # 流派二: 软性注意力
            attention_map = self.sigmoid(edge_map)
        else:
            # 流派一: 硬性门控
            attention_map = edge_map
        
        # 将注意力图与原始特征图相乘
        # (B, C, H, W) * (B, C, H, W) -> (B, C, H, W)
        return x * attention_map

class ChannelAttention(nn.Module):
    """
    论文中描述的通道注意力模块 (Channel Attention Module, CA)。
    这个实现严格遵循了公式(5)的逻辑。
    """
    def __init__(self, in_channels, reduction_ratio=4):
        """
        初始化通道注意力模块。
        :param in_channels: 输入特征图的通道数。
        :param reduction_ratio: 通道缩减率r，用于MLP的瓶颈层。论文中没有明确给出，但16是常用值。
        """
        super(ChannelAttention, self).__init__()
        # 检查缩减率是否合理
        if in_channels <= reduction_ratio:
            # 如果输入通道数本身就很小，直接使用输入通道数的一半或者1作为中间通道
            # 避免降维后通道数为0或负数
            mip_channels = in_channels // 2 if in_channels > 1 else 1
        else:
            mip_channels = in_channels // reduction_ratio
        # 1. Squeeze 操作: 全局平均池化和全局最大池化
        # 这两个操作在forward函数中直接调用F.adaptive_avg_pool2d和F.adaptive_max_pool2d实现
        # 所以这里不需要定义层
        # 2. Shared MLP: 一个共享的多层感知机
        # 使用1x1卷积来实现全连接层，这是CNN中的标准做法
        self.shared_mlp = nn.Sequential(
            # 第一个1x1卷积，对应 W0，用于降维
            nn.Conv2d(in_channels, mip_channels, kernel_size=1, bias=False),
            # ReLU激活函数，对应 δ
            nn.ReLU(inplace=True),
            # 第二个1x1卷积，对应 W1，用于升维
            nn.Conv2d(mip_channels, in_channels, kernel_size=1, bias=False)
        )
        
        # 3. Sigmoid 激活函数，对应 σ
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        """
        前向传播过程。
        :param x: 输入特征图，尺寸 (B, C, H, W)
        :return: 经过通道注意力加权后的特征图，尺寸 (B, C, H, W)
        """
        # 保存原始输入，用于最后的乘法
        original_input = x
        # 获取输入尺寸
        B, C, H, W = x.size()
        # Squeeze 操作
        # 全局平均池化 -> (B, C, 1, 1)
        avg_pool_out = F.adaptive_avg_pool2d(x, (1, 1))
        # 全局最大池化 -> (B, C, 1, 1)
        max_pool_out = F.adaptive_max_pool2d(x, (1, 1))
        # Shared MLP 操作
        # 分别通过共享的MLP
        avg_mlp_out = self.shared_mlp(avg_pool_out)
        max_mlp_out = self.shared_mlp(max_pool_out)
        # Merge 操作: 元素级相加
        merged_out = avg_mlp_out + max_mlp_out
        # Excitation 操作: Sigmoid
        attention_weights = self.sigmoid(merged_out)
        # Reweight 操作: 元素级相乘
        # 利用广播机制 (B, C, 1, 1) -> (B, C, H, W)
        output = original_input * attention_weights
        
        return output


class SobelGate(nn.Module):
    """
    Sobel门控信号生成器。
    
    这个模块封装了从输入特征图生成门控信号的所有逻辑。
    """
    def __init__(self, channel):
        super().__init__()
        self.epsilon = 1e-6
        
        # 定义Sobel滤波器 (不可学习)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).reshape(1, 1, 3, 3)
        
        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        
        self.conv_x.weight.data = sobel_x.repeat(channel, 1, 1, 1)
        self.conv_y.weight.data = sobel_y.repeat(channel, 1, 1, 1)
        
        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False

        self.norm = nn.BatchNorm2d(channel)
        self.act = nn.GELU()

    def forward(self, x):
        # 混合精度稳定性处理
        input_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        
        # 计算梯度
        edges_x = self.conv_x(x_f32)
        edges_y = self.conv_y(x_f32)
        
        # 计算梯度幅值
        gradient_magnitude = torch.sqrt(edges_x.pow(2) + edges_y.pow(2) + self.epsilon)
    
        # 生成门控信号并返回
        gate_signal = self.act(self.norm(gradient_magnitude))
        
        return gate_signal.to(input_dtype)

class Context_Exploration_Block(nn.Module):
    """
    集成了 C2f 结构、GELU 激活函数和 ECA 注意力机制的上下文探索模块 (最终正确版本)。
    """
    def __init__(self, c1, c2, e=0.5):
        super(Context_Exploration_Block, self).__init__()
        assert c1 == c2, "C1 and C2 should be equal for Context_Exploration_Block"
        
        self.c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c_, k=1, s=1)
        self.channels_single = self.c_ // 4
        
        def get_safe_gn_groups(num_channels, prefer=32):
            num_channels = int(num_channels)
            if num_channels == 0: return 1
            for divisor in [prefer, 16, 8, 4, 2, 1]:
                if num_channels % divisor == 0: return divisor
            return 1

        gn_groups_single = get_safe_gn_groups(self.channels_single)
        gn_groups_c = get_safe_gn_groups(self.c_)

        act = nn.GELU

        # 为 y[0] (直连路径) 定义一个 1x1 卷积处理块
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(self.c_, self.c_, 1, 1, 0, bias=False),
            nn.GroupNorm(gn_groups_c, self.c_),
            act()
        )

        # 多分支路径的完整定义
        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.c_, self.channels_single, 1, 1, 0, bias=False),
            nn.GroupNorm(gn_groups_single, self.channels_single), act())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.c_, self.channels_single, 1, 1, 0, bias=False),
            nn.GroupNorm(gn_groups_single, self.channels_single), act())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.c_, self.channels_single, 1, 1, 0, bias=False),
            nn.GroupNorm(gn_groups_single, self.channels_single), act())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.c_, self.channels_single, 1, 1, 0, bias=False),
            nn.GroupNorm(gn_groups_single, self.channels_single), act())

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0, bias=False),
            nn.GroupNorm(gn_groups_single, self.channels_single), act())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1, dilation=1, bias=False),
            nn.GroupNorm(gn_groups_single, self.channels_single), act())

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1, bias=False),
            nn.GroupNorm(gn_groups_single, self.channels_single), act())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 2, dilation=2, bias=False),
            nn.GroupNorm(gn_groups_single, self.channels_single), act())

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2, bias=False),
            nn.GroupNorm(gn_groups_single, self.channels_single), act())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 4, dilation=4, bias=False),
            nn.GroupNorm(gn_groups_single, self.channels_single), act())

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3, bias=False),
            nn.GroupNorm(gn_groups_single, self.channels_single), act())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 8, dilation=8, bias=False),
            nn.GroupNorm(gn_groups_single, self.channels_single), act())

        fusion_in_channels = 4 * self.channels_single
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_channels, self.c_, 1, 1, 0, bias=False),
            nn.GroupNorm(gn_groups_c, self.c_), act(),
            nn.Conv2d(self.c_, self.c_, 3, 1, 1, bias=False),
            nn.GroupNorm(gn_groups_c, self.c_), act()
        )
        
        self.cv2 = Conv(2 * self.c_, c2, k=1, s=1)
        self.eca = ECA(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))

        # 处理路径
        y_process = y[1]
        p1_input = self.p1_channel_reduction(y_process)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(y_process) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(y_process) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(y_process) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4) 
        
        ce_out = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        # 直连路径
        y0_processed = self.shortcut_conv(y[0])
        
        out = self.cv2(torch.cat((y0_processed, ce_out), 1))
        
        return self.eca(out)

class ConcatShuffleConv(nn.Module):
    """
    一个自定义模块，它将两个输入进行拼接(Concat)，然后进行通道混洗(Channel Shuffle)，
    最后通过一个1x1卷积进行降维。
    """
    def __init__(self, c_list, c_out):
        """
        Args:
            c_list (list of int): 两个输入源的通道数列表，例如 [128, 256]。
                                 这个参数将由解析器自动填充。
            c_out (int):          最终输出的通道数。
        """
        super().__init__()
        # 模块内部不需要存储 c_list，因为它只在初始化时使用
        
        # 1. 计算拼接后的总输入通道数
        c_in = sum(c_list)
        
        # 2. 定义1x1的降维卷积层
        #    它的输入通道数是 c_in，输出通道数是 c_out
        self.conv = Conv(c_in, c_out, k=1, s=1)

    def forward(self, x_list):
        """
        Args:
            x_list (list of Tensor): 包含两个输入特征图的列表。
        
        Returns:
            Tensor: 处理后的输出特征图。
        """
        # 1. 拼接
        # x_list 是一个包含了两个张量的列表
        x_concatenated = torch.cat(x_list, dim=1)
        
        # 2. 通道混洗
        # 我们有两个输入源，所以混洗的组数是 2
        x_shuffled = channel_shuffle(x_concatenated, groups=2)
        
        # 3. 1x1卷积降维
        return self.conv(x_shuffled)

class AAttn(nn.Module):
    """
    Area-attention module with the requirement of flash attention.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.

    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)


    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = qkv.view(B, N, self.num_heads, self.head_dim * 3).split(
            [self.head_dim, self.head_dim, self.head_dim], dim=3
        )

        if x.is_cuda and USE_FLASH_ATTN:
            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)
        elif x.is_cuda and not USE_FLASH_ATTN:
            x = sdpa(
                q.permute(0, 2, 1, 3).contiguous(), 
                k.permute(0, 2, 1, 3).contiguous(), 
                v.permute(0, 2, 1, 3).contiguous(), 
                attn_mask=None, 
                dropout_p=0.0, 
                is_causal=False
            )
            x = x.permute(0, 2, 1, 3)
        else:
            q = q.permute(0, 2, 3, 1)
            k = k.permute(0, 2, 3, 1)
            v = v.permute(0, 2, 3, 1)
            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values 
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = (v @ attn.transpose(-2, -1))
            x = x.permute(0, 3, 1, 2)
            v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        x = x + self.pe(v)
        x = self.proj(x)
        return x
    

class ABlock(nn.Module):
    """
    ABlock class implementing a Area-Attention block with effective feature extraction.

    This class encapsulates the functionality for applying multi-head attention with feature map are dividing into areas
    and feed-forward neural network layers.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        area (int, optional): Number of areas the feature map is divided.  Defaults to 1.

    Methods:
        forward: Performs a forward pass through the ABlock, applying area-attention and feed-forward layers.

    Examples:
        Create a ABlock and perform a forward pass
        >>> model = ABlock(dim=64, num_heads=2, mlp_ratio=1.2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """Initializes the ABlock with area-attention and feed-forward layers for faster feature extraction."""
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Executes a forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class A2C2f(nn.Module):  
    """
    A2C2f module with residual enhanced feature extraction using ABlock blocks with area-attention. Also known as R-ELAN

    This class extends the C2f module by incorporating ABlock blocks for fast attention mechanisms and feature extraction.

    Attributes:
        c1 (int): Number of input channels;
        c2 (int): Number of output channels;
        n (int, optional): Number of 2xABlock modules to stack. Defaults to 1;
        a2 (bool, optional): Whether use area-attention. Defaults to True;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1;
        residual (bool, optional): Whether use the residual (with layer scale). Defaults to False;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        e (float, optional): Expansion ratio for R-ELAN modules. Defaults to 0.5;
        g (int, optional): Number of groups for grouped convolution. Defaults to 1;
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to True;

    Methods:
        forward: Performs a forward pass through the A2C2f module.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import A2C2f
        >>> model = A2C2f(c1=64, c2=64, n=2, a2=True, area=4, residual=True, e=0.5)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        # num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
        num_heads = c_ // 32

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)  # optional act=FReLU(c2)

        init_values = 0.01  # or smaller
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else C3k(c_, c_, 2, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        
        # 1. 先计算出模块的主干输出
        output = self.cv2(torch.cat(y, 1))

        # 2. 只有在两个条件都满足时，才执行残差连接
        #    - 条件一：用户想要残差连接 (self.gamma 被创建)
        #    - 条件二：输入和输出的通道数必须相等 (安全检查)
        if self.gamma is not None and x.shape[1] == output.shape[1]:
            return x + self.gamma.view(1, -1, 1, 1) * output
            
        # 3. 如果条件不满足，就直接返回主干输出
        return output
