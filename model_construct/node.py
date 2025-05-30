# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 14:08:38 2025

@author: leo112
"""

import torch
import torch.nn.functional as F

class CommonConfig:
    def __str__(self):
        var_list = {
            k: v for k, v in vars(self).items() if not k.startswith("_") and not callable(v)
        }
        return "{}({})".format(self.__class__.__name__, var_list)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return vars(self) == vars(other)
    
class XY(CommonConfig):
    def __init__(self, x, y):
        self.x = x
        self.y = y

class XYZ(XY):
    def __init__(self, x, y, z):
        super().__init__(x, y)
        self.z = z

class HW(CommonConfig):
    def __init__(self, h, w):
        self.h = h
        self.w = w


class HWC(HW):
    def __init__(self, h, w, c):
        super().__init__(h, w)
        self.c = c

    def __iter__(self):
        return iter((self.h, self.w, self.c))

class OCHW:
    def __init__(self, o, c, h, w):
        self.o = o  # output channels
        self.c = c  # input channels
        self.h = h  # height
        self.w = w  # width

    def __iter__(self):
        return iter((self.o, self.c, self.h, self.w))

    def __repr__(self):
        return f"OCHW(o={self.o}, c={self.c}, h={self.h}, w={self.w})"


class Node(CommonConfig):
    def __init__(self, name: str, op, ifm_shape: HWC, kernel_shape: OCHW, ifm_data: torch.tensor, kernel_data: torch.tensor,
                 stride=None, padding=None, bias=None, scale_mantissa=None, scale_shift=None, bias_dtype=None, psum_dtype=None,
                 ifm_zp=None, ofm_zp=None, dilation=None, ifm_dtype=None, kernel_dtype=None, ofm_dtype=None, activation=None):
        
        self.name = name
        self.op = op
        self.ifm_shape = ifm_shape
        self.kernel_shape = kernel_shape
        self.ifm_data = ifm_data
        self.kernel_data = kernel_data
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.scale_mantissa = scale_mantissa
        self.scale_shift = scale_shift
        self.ofm_data=None
        self.ofm_shape=None
        self.ifm_zp = ifm_zp
        self.ofm_zp=ofm_zp
        self.dilation = dilation
        self.ifm_dtype=ifm_dtype
        self.kernel_dtype=kernel_dtype
        self.ofm_dtype=ofm_dtype
        self.bias_dtype=bias_dtype
        self.activation=activation
        self.psum_dtype=psum_dtype
        self.req_data=None
 
    
    def __hash__(self): # utility func to make sure that Node is hashable
        return hash((self.name, self.op))

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.name == other.name and self.op == other.op
    
    def assign_torch_dtype(dtype):
        if dtype == "int8":
            return torch.int8
        elif dtype == "int16":
            return torch.int16
        elif dtype == "int32":
            return torch.int32
        elif dtype == "uint8":
            return torch.uint8
        elif dtype == "uint16":
            return torch.uint16
        elif dtype == "uint32":
            return torch.uint32
        else:
            raise ValueError("Not supported dtype: {}".format(dtype))
    
    def assign_bitwidth(dtype):
        if dtype == "int8":
            return 8
        elif dtype == "int16":
            return 16
        elif dtype == "int32":
            return 32
        elif dtype == "uint8":
            return 8
        elif dtype == "uint16":
            return 16
        elif dtype == "uint32":
            return 32
        else:
            raise ValueError("Not supported dtype: {}".format(dtype))
        
    def assign_value_range(dtype):
        if dtype == "int8":
            bit=8
            value= 2**(bit-1)
            return (-value, value-1)
        elif dtype == "int16":
            bit=16
            value= 2**(bit-1)
            return (-value, value-1)
        elif dtype == "int32":
            bit=32
            value= 2**(bit-1)
            return (-value, value-1)
        elif dtype == "uint8":
            bit=8
            value= 2**(bit)
            return (0, value)
        elif dtype == "uint16":
            bit=16
            value= 2**(bit)
            return (0, value)
        elif dtype == "uint32":
            bit=32
            value= 2**(bit)
            return (0, value)
        
        
    def conv2d(self):
        # Convert padding from [top, left, bottom, right] → torch format (left, right, top, bottom)
        
        pad = [self.padding[1], self.padding[3], self.padding[0], self.padding[2]]  # (left, right, top, bottom)
        ifm_padded = F.pad(self.ifm_data, pad, mode='constant', value=0)
         
        # kernel is in OCHW (O, C, H, W)
        # Perform convolution
        self.ofm_data = F.conv2d(
            ifm_padded,
            self.kernel_data,
            bias=self.bias,
            stride=(self.stride.y, self.stride.x)
        )

        ofm_int32 = self.ofm_data.to(torch.int32)
        # rounding = 1 << (int(self.scale_shift) - 1)
        rounding=0
        ofm_scaled = (ofm_int32 * int(self.scale_mantissa) + rounding) >> int(self.scale_shift)
        if self.activation is None:
            clipped = ofm_scaled
        elif self.activation=="ReLU":
            clipped = torch.clamp(ofm_scaled, 0, 2**(Node.assign_bitwidth(self.ofm_dtype))-1)
        elif self.activation=="Clamp":
            clipped = torch.clamp(ofm_scaled, -(2**(Node.assign_bitwidth(self.ofm_dtype)-1)-1), 2**(Node.assign_bitwidth(self.ofm_dtype)-1)-1)
        else:
            raise ValueError("Not supported activation: {}".format(self.activation))
        
        self.req_data = clipped.to(Node.assign_torch_dtype(self.ofm_dtype))
        # Convert output tensor shape: (N, C, H, W) → HWC
        _, c, h, w = self.ofm_data.shape
        self.ofm_shape = HWC(h, w, c)  # Custom class

        return self.req_data