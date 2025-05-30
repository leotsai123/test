from macro_gen.isa_def import Dtype
from typing import Tuple
import numpy as np
import torch

class Shape:

    def __init__(self, *args):
        if len(args) == 1:
            if isinstance(args[0], Tuple):
                if len(args[0]) == 2 and all(isinstance(i, int) for i in args[0]):
                    self.h = args[0][0]
                    self.w = args[0][1]
                    # print("height: ", self.h, "width: ", self.w)
            else:
                raise NotImplementedError("THE CONSTRUCTOR IS NOT SUPPORTED")
        elif len(args) == 2:
            if isinstance(args[0], int) and isinstance(args[1], int):
                self.h = args[0]
                self.w = args[1]
            else:
                raise NotImplementedError("THE CONSTRUCTOR IS NOT SUPPORTED")

    def __add__(self, other):
        return Shape(self.h + other.h, self.w + other.w)

    def __sub__(self, other):
        return Shape(self.h - other.h, self.w - other.w)

    def __str__(self) -> str:
        return f"{self.h} x {self.w}"

    def __getitem__(self, index):
        if index == 0:
            return self.h
        elif index == 1:
            return self.w
        else:
            raise NotImplementedError("THE INDEX IS NOT SUPPORTED")

    def get_size(self):
        return self.h * self.w

    def as_tuple(self):
        return (self.h, self.w)
    
def pack_int8_to_int64(values):
    assert len(values) == 8, "fetched input isn't B8 aligned!!!"
    packed = np.int64(0)
    for i, val in enumerate(values):
        byte = np.uint8(val)  # Ensure value is treated as a byte
        packed |= np.int64(byte) << (8 * i)
    return packed

def dtype_to_bytes(dtype):
    if dtype == Dtype.INT8 or dtype == "int8":
        return 1
    elif dtype == Dtype.INT16 or dtype == "int16":
        return 2
    elif dtype == Dtype.INT32 or dtype == "int32":
        return 4
    elif dtype == Dtype.UINT8 or dtype == "uint8":
        return 1
    elif dtype == Dtype.UINT16 or dtype == "uint16":
        return 2
    elif dtype == Dtype.UINT32 or dtype == "uint32":
        return 4
    else:
        raise ValueError("Not supported dtype: {}".format(dtype))
    
def assign_torch_dtype(dtype):
    if dtype == Dtype.INT8 or dtype == "int8":
        return torch.int8
    elif dtype == Dtype.INT16 or dtype == "int16":
        return torch.int16
    elif dtype == Dtype.INT32 or dtype == "int32":
        return torch.int32
    elif dtype == Dtype.UINT8 or dtype == "uint8":
        return torch.uint8
    elif dtype == Dtype.UINT16 or dtype == "uint16":
        return torch.uint16
    elif dtype == Dtype.UINT32 or dtype == "uint32":
        return torch.uint32
    else:
        raise ValueError("Not supported dtype: {}".format(dtype))
    
def assign_np_dtype(dtype):
    if dtype == Dtype.INT8 or dtype == "int8":
        return np.int8
    elif dtype == Dtype.INT16 or dtype == "int16":
        return np.int16
    elif dtype == Dtype.INT32 or dtype == "int32":
        return np.int32
    elif dtype == Dtype.UINT8 or dtype == "uint8":
        return np.uint8
    elif dtype == Dtype.UINT16 or dtype == "uint16":
        return np.uint16
    elif dtype == Dtype.UINT32 or dtype == "uint32":
        return np.uint32
    else:
        raise ValueError("Not supported dtype: {}".format(dtype))
    
def assign_bitwidth(dtype):
    if dtype == Dtype.INT8 or dtype == "int8":
        return 8
    elif dtype == Dtype.INT16 or dtype == "int16":
        return 16
    elif dtype == Dtype.INT32 or dtype == "int32":
        return 32
    elif dtype == Dtype.UINT8 or dtype == "uint8":
        return 8
    elif dtype == Dtype.UINT16 or dtype == "uint16":
        return 16
    elif dtype == Dtype.UINT32 or dtype == "uint32":
        return 32
    else:
        raise ValueError("Not supported dtype: {}".format(dtype))
    
