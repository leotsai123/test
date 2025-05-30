# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 14:09:32 2025

@author: user
"""
import torch
import math

def to_nhcwb8(tensor: torch.Tensor , ic_align: int = 8) -> torch.Tensor:
    """
    Convert NCHW tensor to NHCWB8 layout, with zero padding if C is not divisible by 8.

                 [0, 1, 2, 3]
    Input shape: [N, C, H, W]
    Output shape: [N, H, C//8 or padded, W, 8]

    """
    # print("tensor origin: ", tensor.shape)
    tensor_nhcw = tensor.permute(0,2,1,3)
    N, H, C, W = tensor_nhcw.shape
    padded_C = math.ceil(C / ic_align) * ic_align

    if padded_C != C:
        pad_channels = padded_C - C
        # Pad along channel dimension (dim=1) with zeros
        pad_tensor = torch.zeros((N, H, pad_channels, W), dtype=tensor.dtype, device=tensor.device)
        tensor_nhcw = torch.cat([tensor_nhcw, pad_tensor], dim=2)
    #            [0, 1, 2   , 3, 4]
    # Reshape to [N, H, C//8, 8, W] → permute to [N, H, C//8, W, 8]
    tensor_nhcwb8 = tensor_nhcw.view(N, H, padded_C // 8, 8, W).permute(0, 1, 2, 4, 3).contiguous()

    # print("tensor before flatten: ", tensor_nchwb8.shape)
    # flattend = tensor_nhcwb8.view(-1).shape
    # print("tensor after flatten: ", flattend)
    return tensor_nhcwb8.view(-1)

def to_nchwb8(tensor: torch.Tensor , ic_align: int = 8) -> torch.Tensor:
    """
    Convert NCHW tensor to NCHWB8 layout, with zero padding if C is not divisible by 8.

                 [0, 1, 2, 3]
    Input shape: [N, C, H, W]
    Output shape: [N, H, C//8 or padded, W, 8]

    """
    # print("tensor origin: ", tensor.shape)
    N, C, H, W = tensor.shape
    padded_C = math.ceil(C / ic_align) * ic_align

    if padded_C != C:
        pad_channels = padded_C - C
        # Pad along channel dimension (dim=1) with zeros
        pad_tensor = torch.zeros((N, pad_channels, H, W), dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, pad_tensor], dim=1)

    # Reshape to [N, C//8, 8, H, W] → permute to [N, C//8, H, W, 8]
    tensor_nchwb8 = tensor.view(N, padded_C // 8, 8, H, W).permute(0, 1, 3, 4, 2).contiguous()

    # print("tensor before flatten: ", tensor_nchwb8.shape)
    # flattend = tensor_nchwb8.view(-1).shape
    # print("tensor after flatten: ", flattend)
    return tensor_nchwb8.view(-1)


def to_nchw(tensor: torch.Tensor, original_C: int) -> torch.Tensor:
    """
    Convert from NCHWB8 back to NCHW and optionally trim extra channels.

    Input: [N, Cb, H, W, 8]
    Output: [N, original_C, H, W]
    """
    N, Cb, H, W, B = tensor.shape
    out = tensor.permute(0, 1, 4, 2, 3).contiguous().view(N, Cb * B, H, W)
    
    return out[:, :original_C, :, :]

def to_ochwb8(tensor_ochw: torch.Tensor, oc_align: int = 8) -> torch.Tensor:
    """
    Convert kernel from OHWC to OHWCB8 layout, padding input channels if needed.

    Input shape: [O, C, H, W]
    Output shape: [O//8, padded_C, H, W, 8] → flattened to 1D

    Args:
        kernel_ohwc (torch.Tensor): Kernel tensor in OHWC layout.
        oc_align (int): Input channel alignment (default is 8).

    Returns:
        torch.Tensor: Tensor reshaped to OHWCB8 format and flattened.
    """
    # print("tesnsor origin: ", tensor_ochw.shape)
    O, C, H, W = tensor_ochw.shape
    assert O % 8 == 0, "Output channels must be divisible by 8"

    padded_C = math.ceil(C / oc_align) * oc_align
    if padded_C != C:
        pad_width = padded_C - C
        pad_tensor = torch.zeros((O, pad_width, H, W), dtype=tensor_ochw.dtype, device=tensor_ochw.device)
        tensor_ochw = torch.cat([tensor_ochw, pad_tensor], dim=1)  # pad channel dim

    # Reshape to [O//8, 8, padded_C, H, W] → permute to [O//8, padded_C, H, W, 8]
    tensor_ochw = tensor_ochw.view(O // 8, 8, padded_C, H, W).permute(0, 2, 3, 4, 1).contiguous()
    # print("tensor before flatten: ", tensor_ochw.shape)
    flattend = tensor_ochw.view(-1).shape
    # print("tensor after flatten: ", flattend)
    return tensor_ochw.view(-1)  # Flatten to 1D

    
