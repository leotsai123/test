"""
Created on Sat Apr 19 14:08:38 2025

@author: leo112
"""

from typing import List, Tuple, Dict, Union, Optional
from os import path, makedirs
from model_construct.node import Node, XY
from model_construct.op_attrs import (
    CONV2D_OP
)
import torch
from io import StringIO
import os
import math

# Base directory where your script is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TILING_DIR = os.path.join(BASE_DIR, '..', 'build', 'tilings')
TILING_DIR = os.path.normpath(TILING_DIR)

TILING_DEBUG_DIR = os.path.join(BASE_DIR, '..', 'tiling', 'debug')
TILING_DEBUG_DIR = os.path.normpath(TILING_DEBUG_DIR)

WINDOW_OP = ['CONV2D_OP']
# VRF SRAM constraint
UNI_SRAM_SIZE = 16 * 1024
# Hardware constraint
CIM_COL = 8
CIM_IC = 128
CIM_OC = 32

def dtype_to_bytes(dtype):
    if dtype == "int8":
        return 1
    elif dtype == "int16":
        return 2
    elif dtype == "int32":
        return 4
    elif dtype == "int64":
        return 8
    elif dtype == "uint8":
        return 1
    elif dtype == "uint16":
        return 2
    elif dtype == "uint32":
        return 4
    elif dtype == "uint64":
        return 8
    elif dtype == "float32":
        return 4
    else:
        raise ValueError("Not supported dtype: {}".format(dtype))


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
    

# alias for reading friendly
Coord = Shape

class CallNodeMeta:
    def __init__(self, callnode: Node):
        self.ifm_shape = self.get_ifm_shape(callnode)                                   # node layer height and width -> Shape
        self.p_ifm_shape = self.get_p_ifm_shape(callnode)                               # node layer padded height and width -> Shape
        self.hw_coord = Coord(0, 0)                                                     # height and width of the current ifm tile cursor -> Coord
        self.p_hw_coord = Coord(0, 0)                                                   # height and width of the padded ifm tile in the hardware -> Coord
        self.valid_padding: List[int] = [0, 0, 0, 0]
        self.hw_coord_padding_map: Dict[Tuple[int, int], List[int]] = {}                # padding information of each ifm tile cursor -> Dict[key(tuple), value(list of int)]
        self.i_tile_channel_queue: Dict[Tuple[int, int], Union[List[int], None]] = {}   # maps each ifm tile to its input channel -> Dict[key(tuple), value(list of int)]
        self.i_tile_shape_map: Dict[Tuple[int, int], Tuple[int, int]] = {}              # maps each ifm tile coord to its shape -> Dict[key(tuple), value(tuple)]
        self.i_tile_offset_map: Dict[Tuple[int, int], List[int]] = {}                   # maps each ifm tile coord to its mem offset -> Dict[key(tuple), value(list of int)]
        self.o_tile_channel_queue: Dict[Tuple[int, int], Union[List[int], None]] = {}   # maps each ofm tile to its input channel -> Dict[key(tuple), value(list of int)]
        self.o_tile_shape_map: Dict[Tuple[int, int], Tuple[int, int]] = {}              # maps each ofm tile coord to its shape -> Dict[key(tuple), value(tuple)]
        self.o_tile_offset_map: Dict[Tuple[int, int], List[int]] = {}                   # maps each ofm tile coord to its mem offset -> Dict[key(tuple), value(list of int)]
        self.from_dram: bool = False                                                    # set whether this node is from dram or not -> bool
        self.to_dram: bool = False                                                      # set whether this node is to dram or not -> bool
        self.weight_offset: List[int] = []                                              # weight offset regarding to each tiled ofm channel group, no need to create dict since it is reused across ifm tiles
        self.bias_offset: List[int] = []                                                # tiled bias offset 
        self.scale_offset: List[int] = []                                               # tiled scale offset
        self.i_coord_o_coord_map: Dict[Tuple[int, int], Tuple[int, int]] = {}           # maps each ifm tile coord to its ofm tile coord -> Dict[key(tuple), value(tuple)]
        self.i_coord_p_i_coord_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
    def reset(self):                       
        self.hw_coord = Coord(0, 0)                                                    
        self.p_hw_coord = Coord(0, 0)                                                  
        self.valid_padding: List[int] = [0, 0, 0, 0]
        self.hw_coord_padding_map: Dict[Tuple[int, int], List[int]] = {}               
        self.i_tile_channel_queue: Dict[Tuple[int, int], Union[List[int], None]] = {}  
        self.i_tile_shape_map: Dict[Tuple[int, int], Tuple[int, int]] = {}             
        self.i_tile_offset_map: Dict[Tuple[int, int], List[int]] = {}                  
        self.o_tile_channel_queue: Dict[Tuple[int, int], Union[List[int], None]] = {}  
        self.o_tile_shape_map: Dict[Tuple[int, int], Tuple[int, int]] = {}             
        self.o_tile_offset_map: Dict[Tuple[int, int], List[int]] = {}                  
        self.from_dram: bool = False                                                   
        self.to_dram: bool = False                                                     
        self.tile_dram_offset: List[int] = []                                          
        self.weight_offset: List[int] = []                                             
        self.bias_offset: List[int] = []                                               
        self.scale_offset: List[int] = []                                              

    def get_ifm_shape(self, callnode: Node):
        return Shape(callnode.ifm_shape.h, callnode.ifm_shape.w)
    
    def get_p_ifm_shape(self, callnode: Node):
        # padding = [top, left, bottom, right]
        return Shape(
            self.ifm_shape.h + callnode.padding[0] + callnode.padding[2],
            self.ifm_shape.w + callnode.padding[1] + callnode.padding[3]
        )
    
    def get_padding(self, callnode: Node, p_i_tile_shape: Shape):
        """
        Get the padding position of the current padded ifm tile.

        Parameters
        ----------
            callnode and the padded ifm tile

        Returns
        -------
            padding position
        """
        if hasattr(callnode, "padding") and isinstance(callnode.padding, list) and len(callnode.padding) == 4:
            # print("has padding attr")
            padding = callnode.padding # if it is conv2d, then [1,1,1,1]
            padding_trigger = [
                self.p_hw_coord.h == 0,
                self.p_hw_coord.w == 0,
                self.p_hw_coord.h + p_i_tile_shape.h >= self.p_ifm_shape.h,
                self.p_hw_coord.w + p_i_tile_shape.w >= self.p_ifm_shape.w
            ]
            padding = [
                padding[0] if padding_trigger[0] else 0,
                padding[1] if padding_trigger[1] else 0,
                padding[2] if padding_trigger[2] else 0,
                padding[3] if padding_trigger[3] else 0
            ]
        else:
            padding = [0, 0, 0, 0]

        self.valid_padding = padding
        return padding
    
    def update_p_coord(self, p_i_tile_shape: Shape, next_row: bool, overlap: Shape):
        """
        Based on the padded ifm tile shape, calculate the next padded ifm tile coord.

        Parameters
        ----------
            p_i_tile_shape, next_row, overlap
            
        Returns
        -------
            No return values, update callnode's meta data
        """

        self.i_coord_p_i_coord_map[(self.hw_coord.h, self.hw_coord.w)] = (
            self.p_hw_coord.h,
            self.p_hw_coord.w,
        )

        if not next_row:
            self.p_hw_coord.w = self.p_hw_coord.w + p_i_tile_shape.w - overlap.w
        else:
            self.p_hw_coord.h = self.p_hw_coord.h + p_i_tile_shape.h - overlap.h
            self.p_hw_coord.w = 0
        
    def update_padding(self, valid_padding: List[int]):
        """
        Record the valid padding position based on the current hw_coord

        Parameters
        ----------
            valid_padding
            
        Returns
        -------
            No return values, update callnode's meta data
        """
        
        self.hw_coord_padding_map[(self.hw_coord.h, self.hw_coord.w)] = valid_padding
        pass

    def update_coord(self, i_tile_shape: Shape, next_row: bool, overlap: Shape, o_tile_coord: Coord):
        """
        1. Based on the current hw_coord, insert it as a key into the meta data dicts. i.e., i_tile_channel_queue, i_tile_shape_map, i_tile_offset_map and i_coord_o_coord_map
        2. Based on the ifm tile shape, ofm tile corrd, next_row and overlap information, calculate the next ifm tile coord.
        
        Parameters
        ----------
            i_tile_shape, next_row, overlap, o_tile_coord
            
        Returns
        -------
            No return values, update callnode's meta data
        """
        self.i_tile_channel_queue[(self.hw_coord.h, self.hw_coord.w)] = []
        self.i_tile_shape_map[(self.hw_coord.h, self.hw_coord.w)] = i_tile_shape.as_tuple()
        self.i_tile_offset_map[(self.hw_coord.h, self.hw_coord.w)] = []
        self.i_coord_o_coord_map[(self.hw_coord.h, self.hw_coord.w)] = (
            o_tile_coord.h,
            o_tile_coord.w,
        )

        if not next_row:
            self.hw_coord.w = self.hw_coord.w + i_tile_shape.w - overlap.w
        else:
            self.hw_coord.h = self.hw_coord.h + i_tile_shape.h - overlap.h
            self.hw_coord.w = 0

    def get_hw_coord(self):
        return self.hw_coord
    
    def get_p_hw_coord(self):
        return self.p_hw_coord

    @staticmethod
    def cal_weight_offset(callnode: Node, cg: int, ch_per_group: int) -> int:
        """
        Calculate the weight offset based on the callnode's weight and ofm channel group size.
        
        Parameters
        ----------
            callnode, cg(channel group)
            
        Returns
        -------
            weight_offset
        """
        # OCHW
        # Note: addr(OCHWB8) = addr(OCHW/8)
        if callnode.op == CONV2D_OP:
            weight_shape = callnode.kernel_shape
            weight_dtype = callnode.kernel_dtype
            data_byte = dtype_to_bytes(weight_dtype)

            weight_offset = (
               weight_shape.h * weight_shape.w * weight_shape.c * cg * CIM_OC * data_byte
            )

            if weight_offset > (
                weight_shape.o * weight_shape.h * weight_shape.w * weight_shape.c * data_byte
            ):
                raise ValueError(f"[TILING] ERROR THE WEIGHT OFFSET {weight_offset} IS TOO BIG")
            return weight_offset
        
    @staticmethod
    def cal_bias_offset(callnode: Node, cg: int) -> int:
        """
        Calculate the bias offset based on the callnode's bias and ofm channel group size.
        
        Parameters
        ----------
            callnode, cg(channel group)
            
        Returns
        -------
            bias_offset
        """
        op_name = callnode.op
        if op_name == CONV2D_OP:
            if hasattr(callnode, "bias"):
                bias_shape = callnode.ofm_shape.c # OC
                bias_dtype = callnode.bias_dtype
                data_byte = dtype_to_bytes(bias_dtype)

                bias_offset = (
                    cg * CIM_OC * data_byte
                )

                if bias_offset > bias_shape * dtype_to_bytes(bias_dtype):
                    raise ValueError(f"{op_name=}, {cg=}, {bias_offset=}, {bias_shape=}")
                return bias_offset
            else: return -1
        
    @staticmethod
    def cal_scale_offset(callnode: Node, cg: int) -> int:
        """
        Calculate the scale offset based on the callnode's bias and ofm channel group size.
        
        Parameters
        ----------
            callnode, cg(channel group)
            
        Returns
        -------
            scale_offset
        """
        op_name = callnode.op
        if op_name == CONV2D_OP:
            # Support per channel scale for now, and the scale is the same across all channels
            scale_offset = -1
            return scale_offset

            # TODO: Support scale mode with scale values are different across all channels
            # scale_shape = callnode.ofm_shape.c # OC
            # scale_dtype = callnode.scale_mantissa.dtype 
            # data_byte = dtype_to_bytes(scale_dtype)

            # scale_offset = (
            #     cg * CIM_OC * data_byte
            # )

            # if scale_offset > scale_shape * dtype_to_bytes(scale_dtype):
            #     raise ValueError(f"{op_name=}, {cg=}, {scale_offset=}, {scale_shape=}")
            # return scale_offset
        else:
            raise ValueError(f"op_name={callnode.op} is not supported")

    @staticmethod
    def cal_i_tile_offset(callnode: Node, i_tile_coord: Union[Coord, Tuple[int, int]], ch_per_group: int) -> int:
        """
        Calculate the ifm tile offset from DRAM, data format is NHCWB8
        
        Parameters
        ----------
            callnode, i_tile_coord, ch_per_group
            
        Returns
        -------
            i_tile_offset
        """

        if isinstance(i_tile_coord, tuple):
            i_tile_coord = Coord(*i_tile_coord)  # Convert tuple to Coord

        if callnode.op == CONV2D_OP:
            op_name = callnode.op
            ifm_shape = callnode.ifm_shape # NHCWB8
            ifm_dtype = callnode.ifm_dtype
            data_byte = dtype_to_bytes(ifm_dtype)
            ifm_h, ifm_w, ifm_c = ifm_shape
            p_ifm_c = math.ceil(ifm_c/ch_per_group) * ch_per_group 

            if ifm_c < ch_per_group:
                # TODO: pad the input channel now for simplicity, still needs to consider vpu
                # @NOTE: vpu sees byte address, padding problems may be solved by wrapper
                i_tile_offset = (
                    i_tile_coord.h * ifm_w * ch_per_group + i_tile_coord.w * ch_per_group
                ) * data_byte
            else:
                i_tile_offset = (
                    i_tile_coord.h * ifm_w * ifm_c + i_tile_coord.w * ch_per_group
                ) * data_byte
            if i_tile_offset > ifm_h * ifm_w * p_ifm_c * data_byte:
                raise ValueError(
                    f"{op_name=}, {i_tile_coord.h=}, {i_tile_coord.w=}, {i_tile_offset=}, {ifm_h=}, {ifm_w=}, {ifm_c=}"
                )
            return i_tile_offset

    @staticmethod
    def cal_o_tile_offset(callnode: Node, cg: int, o_tile_coord: Union[Coord, Tuple[int,int]], ch_per_group: int) -> int:
        """
        Calculate the ofm tile offset to DRAM, data format is NHCWB8

        Data layout: NHCWB8
            = h_coord * ofm_c * ofm_w + och group * CIM_OC * ofm_w + w_coord * B8
        
        Parameters
        ----------
            callnode, o_tile_coord, cg
            
        Returns
        -------
            o_tile_offset
        """

        if isinstance(o_tile_coord, tuple):
            o_tile_coord = Coord(*o_tile_coord)  # Convert tuple to Coord

        if callnode.op == CONV2D_OP:
            op_name = callnode.op
            ofm_shape = callnode.ofm_shape 
            ofm_dtype = callnode.ofm_dtype
            data_byte = dtype_to_bytes(ofm_dtype)
            ofm_h, ofm_w, ofm_c = ofm_shape

            o_tile_offset = (
                o_tile_coord.h * ofm_c * ofm_w +
                cg * CIM_OC * ofm_w +
                o_tile_coord.w * min(ofm_c - cg * CIM_OC, ch_per_group)
            )
            
            if o_tile_offset > ofm_h * ofm_w * ofm_c * data_byte:
                raise ValueError(
                    f"{op_name=}, {o_tile_coord.h=}, {o_tile_coord.w=}, {o_tile_offset=}, {ofm_h=}, {ofm_w=}, {ofm_c=}"
                )
            return o_tile_offset

class TileAttrsBuilder:

    def __init__(self, ch_per_group):
        self.callnode_overlap_map: Dict[Node, Shape] = {}       # actually currently it can only process one node at a time
        self.callnode_meta_map: Dict[Node, CallNodeMeta] = {}   # actually currently it can only process one node at a time
        self.ch_per_group = ch_per_group
        self.debug_stream = StringIO()

    def reset(self):
        self.callnode_overlap_map = {}
        self.callnode_meta_map = {}

    def init_callnode_meta(self, callnode: Node):
        if callnode not in self.callnode_meta_map:
            self.callnode_meta_map[callnode] = CallNodeMeta(callnode)
    
    def get_overlap(self, callnode: Node):
        if callnode.op == CONV2D_OP:
            return Shape(
                callnode.kernel_shape.h - callnode.stride.y,
                callnode.kernel_shape.w - callnode.stride.x,
            )
        else:
            return Shape(0, 0)
        
    def infer_overlap(self, callnode: Node):
        overlap = self.get_overlap(callnode)
        self.callnode_overlap_map[callnode] = overlap
        
    def infer_i_tile_shape(self, callnode: Node, o_tile_shape: Shape):
        """
        Infer the shape of the ifm tile from the layer and the output tile shape.

        Formula
        -------
            input_pixel = stride * output_pixel - stride + kernel_size

        Parameters
        ----------
            callnode and the output tile shape
        Returns
        -------
            the shape of the required input tile
        """
        stride = callnode.stride
        kernel_shape = callnode.kernel_shape
        
        i_tile_shape = Shape(
            stride.y * o_tile_shape.h + kernel_shape.h - stride.y,
            stride.x * o_tile_shape.w + kernel_shape.w - stride.x,
        )
        return i_tile_shape

    def is_valid_window_size(self, callnode: Node, window_size: Shape) -> bool:
        """
        Check if the window size is valid for the given ifm tile.
        You can view window as a expandable tile, it can sometimes be invalid if the stride is 2

        Parameters
        ----------
            callnode and the window size(p_i_tile_shape)

        Returns
        -------
            True if the window size is valid, False otherwise
        """
        stride = callnode.stride
        kernel_shape = callnode.kernel_shape

        if (window_size.h - kernel_shape.h + stride.y) % stride.y == 0 and (
            window_size.w - kernel_shape.w + stride.x
        ) % stride.x == 0 == 0:
            return True
        else:
            print("invalid window size !!!")
            return False
        
    def remove_padding(self, callnode: Node, p_i_tile_shape: Shape, valid_padding: List[int], flag: bool):
        """
        Removes the padding pixels from the padded input tile shape.

        Parameters
        ----------
            callnode and the padded input tile shape, flag indicate the initial finding maximum i_tile_shape process

        Returns
        -------
            the shape of the input tile after removing padding
        """
        if callnode.op == CONV2D_OP and not flag:
            i_tile_shape = Shape(
                p_i_tile_shape.h - valid_padding[0] - valid_padding[2],
                p_i_tile_shape.w - valid_padding[1] - valid_padding[3],
            )
            return i_tile_shape
        else: return p_i_tile_shape

    def is_valid_tile_size(self, callnode: Node, i_tile_shape: Shape, o_tile_shape: Shape) -> bool:
        """
        Checks whether the ifm tile and ofm tile size has exceed the memory space
        ofm_tile size is calculated with psum data since we use unified memory space

        Parameters
        ----------
            callnode and ifm/ofm tile shape

        Returns
        -------
            True if the memory space is valid, False otherwise
        """
        if callnode.op == CONV2D_OP:
            _, _, ifm_c = callnode.ifm_shape
            ifm_dtype = callnode.ifm_dtype
            _, _, ofm_c = callnode.ofm_shape
            psum_dtype = callnode.psum_dtype
            bias_dtype = callnode.bias_dtype
            ofm_dtype = callnode.ofm_dtype

            p_ifm_c = math.ceil(ifm_c/self.ch_per_group) * self.ch_per_group

            i_sram_size = i_tile_shape.h * i_tile_shape.w * p_ifm_c * dtype_to_bytes(ifm_dtype)
            if ofm_c >= CIM_OC:
                psum_sram_size = o_tile_shape.h * o_tile_shape.w * CIM_OC * dtype_to_bytes(psum_dtype)
                # ofm_sram_size =  o_tile_shape.h * o_tile_shape.w * CIM_OC * dtype_to_bytes(ofm_dtype)
            else:
                psum_sram_size= o_tile_shape.h * o_tile_shape.w * ofm_c * dtype_to_bytes(psum_dtype)
                # ofm_sram_size =  o_tile_shape.h * o_tile_shape.w * ofm_c * dtype_to_bytes(ofm_dtype)
            bias_sram_size = CIM_OC * dtype_to_bytes(bias_dtype)

            self.debug_stream.write(f"psum_sram_size: {psum_sram_size/1024} KB\n")
            # self.debug_stream.write(f"ofm_sram_size: {ofm_sram_size/1024} KB\n")
            self.debug_stream.write(f"i_sram_size: {i_sram_size/1024}KB\n")
            self.debug_stream.write(f"bias_sram_size: {bias_sram_size/1024}KB\n")
            self.debug_stream.write(f"psum_sram_size + i_sram_size + bias_sram_size: {(psum_sram_size+i_sram_size+bias_sram_size)/1024}KB\n")
            is_valid_size = True if (psum_sram_size + i_sram_size + bias_sram_size) <= UNI_SRAM_SIZE else False
            return is_valid_size

            # if ofm_c > CIM_OC:
            #     o_sram_size = o_tile_shape.h * o_tile_shape.w * CIM_OC * dtype_to_bytes(ofm_dtype)
            #     # print("o_sram_size: ", o_sram_size//1024)
            #     is_o_sram_valid = (
            #         o_tile_shape.h * 
            #         o_tile_shape.w * 
            #         CIM_OC * 
            #         dtype_to_bytes(ofm_dtype) * 4 # This 4 indicate psum space
            #         <= O_SRAM_SIZE
            #     )
            # else:
            #     o_sram_size = o_tile_shape.h * o_tile_shape.w * ofm_c * dtype_to_bytes(ofm_dtype)
            #     # print("o_sram_size: ", o_sram_size//1024)
            #     is_o_sram_valid = (
            #         o_tile_shape.h * 
            #         o_tile_shape.w * 
            #         ofm_c * 
            #         dtype_to_bytes(ofm_dtype)
            #         <= O_SRAM_SIZE
            #     )
            
            # if ifm_c < CIM_IC:
            #     i_sram_size = i_tile_shape.h * i_tile_shape.w * CIM_IC * dtype_to_bytes(ifm_dtype)
            #     # print("i_sram_size: ", i_sram_size//1024)
            #     is_i_sram_valid = (
            #         i_tile_shape.h * 
            #         i_tile_shape.w * 
            #         CIM_IC * 
            #         dtype_to_bytes(ifm_dtype)
            #         <= I_SRAM_SIZE
            #     )
            # else:
            #     i_sram_size = i_tile_shape.h * i_tile_shape.w * ifm_c * dtype_to_bytes(ifm_dtype)
            #     # print("i_sram_size: ", i_sram_size//1024)
            #     is_i_sram_valid = (
            #         i_tile_shape.h * 
            #         i_tile_shape.w * 
            #         ifm_c * 
            #         dtype_to_bytes(ifm_dtype)
            #         <= I_SRAM_SIZE
            #     )
            # return is_i_sram_valid and is_o_sram_valid
        else:
            raise NotImplementedError
    
    def forward(self, callnode: Node, o_tile_shape: Shape) -> Union[bool, None]:
        """
        Forward path:
          1. check the window size is valid e.g. window size can be convered by kernel with stride
          2. check the corresponding SRAM usage is valid, it'll check both OSRAM and ISRAM
        
        Parameters
        ----------
            callnode and ofm tile shape

        Returns
        -------
            True if the memory space and window size is valid, False otherwise
        """

        self.debug_stream.write(f"forward function:\n")

        # calculate the input tile shape with padding
        p_i_tile_shape = self.infer_i_tile_shape(callnode, o_tile_shape)
        self.debug_stream.write(f"p_i_tile_shape: {p_i_tile_shape}\n")

        # check the window size, it can be convered by kernel with stride > 1, which can be invalid sometimes
        is_valid_window_size = self.is_valid_window_size(callnode, p_i_tile_shape)
        self.debug_stream.write(f"is_valid_window_size: {is_valid_window_size}\n")

        # use the input tile shape with padding to get padding position
        valid_padding = self.callnode_meta_map[callnode].get_padding(callnode, p_i_tile_shape)
        self.debug_stream.write(f"valid_padding: {valid_padding}\n")

        # remove padding for next op to calculate the unpadded input tile shape with padding information
        is_initial = True # Set this flag to indicate that it is currently finding the maximum tile size
        i_tile_shape = self.remove_padding(callnode, p_i_tile_shape, valid_padding, is_initial)
        self.debug_stream.write(f"i_tile_shape: {i_tile_shape}\n")

        # check the SRAM usage
        is_valid_tile_size = self.is_valid_tile_size(callnode, i_tile_shape, o_tile_shape)
        self.debug_stream.write(f"is_valid_tile_size: {is_valid_tile_size}\n")
        
        if not is_valid_window_size or not is_valid_tile_size:
            return False
        else:
            # Implement only one node for now
            return True

    def infer_max_out_tile_shape(self, callnode: Node):
        """
        Use forward as a basic function to find valid ifm/ofm tile shape based on the memory constraint
        iteratively expand the tile window size to find the max valid tile shape
        
        Parameters
        ----------
            callnode

        Returns
        -------
            Return the max valid tile shape
        """
        o_tile_shape = Shape(1,1)
        n_o_tile_shape = o_tile_shape + Shape(1,1)
        is_valid = self.forward(callnode, o_tile_shape)
        if (not is_valid) :
            raise ValueError("Infer max out tile shape failed at output shape is (1, 1)")
        else:
            while True:
                # print("n_o_tile_shape", n_o_tile_shape)
                n_is_valid = self.forward(callnode, n_o_tile_shape)
                if is_valid and not n_is_valid:
                    break
                else:
                    is_valid = n_is_valid
                    o_tile_shape += Shape(1,1)
                    n_o_tile_shape += Shape(1,1)
            return o_tile_shape 
        
    def set_from_dram_flag(self, callnode: Node):
        """
        Set the flag to indicate if the input of this current node is from DRAM or not
        Supports only one node for now

        Parameters
        ----------
            callnode

        Returns
        -------
            Return from dram flag
        """
        from_dram = True
        self.callnode_meta_map[callnode].from_dram = from_dram
    
    def set_to_dram_flag(self, callnode: Node):
        """
        Set the flag to indicate if the output of this current node is to DRAM or not
        Supports only one node for now

        Parameters
        ----------
            callnode

        Returns
        -------
            Return to dram flag
        """
        self.callnode_meta_map[callnode].to_dram = True

    def forward_update(self, callnode: Node, next_row: bool, o_tile_coord: Coord, o_tile_shape: Shape):
        """
        Updates callnode's meta data after each o_tile_shape is determined

        Parameters
        ----------
            callnode, next_row, o_tile_coord, o_tile_shape

        Returns
        -------
            No return values, just meta data will be updated during the process.
        """

        # print("o_tile_coord: ", o_tile_coord)
        # print("o_tile_shape: ", o_tile_shape)
        
        # calculate the input tile shape with padding
        p_i_tile_shape = self.infer_i_tile_shape(callnode, o_tile_shape)
        self.debug_stream.write(f"\t\tp_i_tile_shape: {p_i_tile_shape}\n")
        
        # No deed to check the window size since it has been checked in forward pass

        # use the input tile shape with padding to get padding position
        valid_padding = self.callnode_meta_map[callnode].get_padding(callnode, p_i_tile_shape)
        self.debug_stream.write(f"\t\tvalid_padding: {valid_padding}\n")

        # remove padding for next op to calculate the input tile shape with padding
        is_initial = False # Set this flag to flase, meaning we are not in the tiling process. maximum tile size is known.
        i_tile_shape = self.remove_padding(callnode, p_i_tile_shape, valid_padding, is_initial)
        self.debug_stream.write(f"\t\ti_tile_shape: {i_tile_shape}\n")

        # Lookup overlap information stored in the initial infer_overlap function for calculation of the next tile coord
        overlap = self.callnode_overlap_map[callnode]
        self.debug_stream.write(f"\t\toverlap: {overlap}\n")

        # Stores ofm tile shape as value into ofm tile coord key
        self.callnode_meta_map[callnode].o_tile_shape_map[
            o_tile_coord.as_tuple()
        ] = o_tile_shape.as_tuple()
        # Stores ofm tile coord key into the o_tile_offset_map dict, o_tile_offset_map value is calulated after
        self.callnode_meta_map[callnode].o_tile_offset_map[
            o_tile_coord.as_tuple()
        ] = []
        # Stores ofm tile coord key into the o_tile_offset_map dict, o_tile_offset_map value is calulated after
        self.callnode_meta_map[callnode].o_tile_offset_map[
            o_tile_coord.as_tuple()
        ] = []
        # Stores ofm tile coord key into the o_tile_channel_queue dict, o_tile_channel_queue value is calulated after
        self.callnode_meta_map[callnode].o_tile_channel_queue[
            o_tile_coord.as_tuple()
        ] = []
        
        # before update
        self.debug_stream.write(f"\t\tbefore update\n")
        self.debug_stream.write(f"\t\t-------------\n")
        self.debug_stream.write(f"\t\tp_hw_coord: {self.callnode_meta_map[callnode].p_hw_coord}\n")
        self.debug_stream.write(f"\t\thw_coord: {self.callnode_meta_map[callnode].hw_coord}\n")

        i_tile_coord = self.callnode_meta_map[callnode].get_hw_coord()
        self.callnode_meta_map[callnode].i_coord_o_coord_map[i_tile_coord] = o_tile_coord

        self.debug_stream.write(f"\t\ti_tile_coord: {i_tile_coord} is mapped to o_tile_coord: {o_tile_coord}\n")
        # update tile coord with padding -> next tile information
        self.callnode_meta_map[callnode].update_p_coord(p_i_tile_shape, next_row, overlap)
        # record the valid padding -> current tile information
        self.callnode_meta_map[callnode].update_padding(valid_padding)
        # update tile coord without padding -> next tile information
        self.callnode_meta_map[callnode].update_coord(i_tile_shape, next_row, overlap, o_tile_coord)
        # update i_coord_o_coord_map -> next tile information
        # i_tile_coord = self.callnode_meta_map[callnode].get_hw_coord()
        # self.callnode_meta_map[callnode].i_coord_o_coord_map[i_tile_coord] = o_tile_coord

        self.debug_stream.write(f"\t\tafter update\n")
        self.debug_stream.write(f"\t\t------------\n")
        self.debug_stream.write(f"\t\tp_hw_coord: {self.callnode_meta_map[callnode].p_hw_coord}\n")
        self.debug_stream.write(f"\t\thw_coord: {self.callnode_meta_map[callnode].hw_coord}\n")
        


    def hw_tiling(self, callnode: Node, max_output_tile_shape: Shape):
        """
        This function is called after the maximum output tile shape is determined.
        Based on the maximum output tile shape, it will traverse the whole ifm using max_output_tile_shape as a bigger window.
        This window only consider height and width dimension.
        During the process, callnode's meta data will be updated accordingly.

        Parameters
        ----------
            callnode and max_output_tile_shape.

        Returns
        -------
            No return values, just meta data will be updated during the process.
        """
        
        self.debug_stream.write(f"hw_tiling starts!!!\n")
        self.debug_stream.write(f"-------------------\n")

        ofm_h, ofm_w = callnode.ofm_shape.h, callnode.ofm_shape.w
        o_tile_coord = Coord(0,0)
        o_tile_shape = Shape(0,0)
        next_row = False
        while o_tile_coord.h < ofm_h:
            remain_h = ofm_h - o_tile_coord.h
            if remain_h > max_output_tile_shape.h:
                o_tile_shape.h = max_output_tile_shape.h
            else:
                o_tile_shape.h = remain_h
            while o_tile_coord.w < ofm_w:
                remain_w = ofm_w - o_tile_coord.w
                if remain_w > max_output_tile_shape.w:
                    o_tile_shape.w = max_output_tile_shape.w
                    next_row = False
                else:
                    next_row = True
                    o_tile_shape.w = remain_w
                # call the forward_update function will update the callnode's meta data

                self.debug_stream.write(f"\to_tile_coord: {o_tile_coord}\n")
                self.debug_stream.write(f"\to_tile_shape: {o_tile_shape}\n")
                self.debug_stream.write(f"\tcall forward_update\n")
                self.forward_update(callnode, next_row, o_tile_coord, o_tile_shape)
                self.debug_stream.write(f"\t---------next tile---------\n")
                o_tile_coord.w += o_tile_shape.w
            o_tile_coord.w = 0
            o_tile_coord.h+=o_tile_shape.h
    
    def depth_tiling(self, callnode: Node):
        """
        Called after hw_tiling function, this function loops over each ifm tile, and get each ofm tile accordingly,
        based on the ofm tile information, loops over ofm channel dimension to calculate data offset.
        Note: whole ifm tile input channel is appended during each iteration of ofm group, and the remainder of ofm_c as well.

        Parameters
        ----------
            callnode.

        Returns
        -------
            No return values, just meta data will be updated during the process.
        """
        self.debug_stream.write(f"depth_tiling starts!!!\n")
        if callnode.op == CONV2D_OP:
            ifm_c = callnode.ifm_shape.c
            ofm_c = callnode.ofm_shape.c
            oc_groups = ofm_c // CIM_OC
            oc_remain = ofm_c % CIM_OC

            callnode_meta = self.callnode_meta_map[callnode]
            for i_tile_coord in callnode_meta.i_tile_channel_queue.keys(): # for each ifm tile coord in all ifm tiles
                o_tile_coord = callnode_meta.i_coord_o_coord_map[i_tile_coord] # get the corresponding ofm tile coord regarding the ifm tile coord
                
                self.debug_stream.write(f"\ti_tile_coord: {i_tile_coord}\n")
                self.debug_stream.write(f"\to_tile_coord: {o_tile_coord}\n")
                self.debug_stream.write(f"\ti_tile_shape: {callnode_meta.i_tile_shape_map[i_tile_coord]}\n")
                self.debug_stream.write(f"\to_tile_shape: {callnode_meta.o_tile_shape_map[o_tile_coord]}\n")
                self.debug_stream.write(f"\toc_groups: {oc_groups}\n")
                self.debug_stream.write(f"\toc_remain: {oc_remain}\n")
                self.debug_stream.write(f"\tentering oc_groups iteration\n")


                if oc_groups > 0:
                    for cg in range(oc_groups): # for each channel group in all ofm channel groups
                        self.debug_stream.write(f"\t\t-----cg-----: {cg}\n")

                        callnode_meta.o_tile_channel_queue[o_tile_coord].append(CIM_OC) # append the ofm channel group to the ofm tile coord's channel queue
                        callnode_meta.i_tile_channel_queue[i_tile_coord].append(ifm_c) # append all ifm channel to the ifm tile coord's channel queue
                        
                        self.debug_stream.write(f"\t\tappend CIM_OC: {CIM_OC} to o_tile_channel_queue @ cg: {cg}\n")
                        self.debug_stream.write(f"\t\tappend ifm_c: {ifm_c} to i_tile_channel_queue @ cg: {cg}\n")

                        weight_offset = CallNodeMeta.cal_weight_offset(callnode, cg, self.ch_per_group)
                        callnode_meta.weight_offset.append(weight_offset)
                        self.debug_stream.write(f"\t\tweight_offset: {weight_offset}\n")

                        bias_offset = CallNodeMeta.cal_bias_offset(callnode, cg)
                        callnode_meta.bias_offset.append(bias_offset)
                        self.debug_stream.write(f"\t\tbias_offset: {bias_offset}\n")

                        scale_offset = CallNodeMeta.cal_scale_offset(callnode, cg)
                        callnode_meta.scale_offset.append(scale_offset)
                        self.debug_stream.write(f"\t\tscale_offset: {scale_offset}\n")


                        if callnode_meta.from_dram:
                            i_tile_offset = CallNodeMeta.cal_i_tile_offset(
                                callnode, i_tile_coord, self.ch_per_group
                            )
                             # NOTE: I think there might have a information to record how many groups of (ifm_c//ch_per_group)
                            callnode_meta.i_tile_offset_map[i_tile_coord].append(i_tile_offset)
                            self.debug_stream.write(f"\t\ti_tile_offset: {i_tile_offset}\n")   
                           
                        else:
                            self.debug_stream.write(f"\t\ti_tile_offset: -1\n") 
                            callnode_meta.i_tile_offset_map[i_tile_coord].append(-1)

                        if callnode_meta.to_dram:
                            o_tile_offset = CallNodeMeta.cal_o_tile_offset(
                                callnode, cg, o_tile_coord, self.ch_per_group
                            )
                            callnode_meta.o_tile_offset_map[o_tile_coord].append(o_tile_offset)
                            self.debug_stream.write(f"\t\to_tile_offset: {o_tile_offset}\n")  
                        else:
                            self.debug_stream.write(f"\t\to_tile_offset: -1\n") 
                            callnode_meta.o_tile_offset_map[o_tile_coord].append(-1)

                if oc_remain > 0:
                    callnode_meta.o_tile_channel_queue[o_tile_coord].append(oc_remain)
                    callnode_meta.i_tile_channel_queue[i_tile_coord].append(ifm_c)

                    self.debug_stream.write(f"\t\tappend CIM_OC: {CIM_OC} to o_tile_channel_queue @ oc_remain: {oc_remain}\n")
                    self.debug_stream.write(f"\t\tappend ifm_c: {ifm_c} to i_tile_channel_queue @ oc_remain: {oc_remain}\n")

                    weight_offset = CallNodeMeta.cal_weight_offset(callnode, oc_groups)
                    callnode_meta.weight_offset.append(weight_offset)
                    self.debug_stream.write(f"\t\tweight_offset: {weight_offset}\n")

                    bias_offset = CallNodeMeta.cal_bias_offset(callnode, oc_groups)
                    callnode_meta.bias_offset.append(bias_offset)
                    self.debug_stream.write(f"\t\tbias_offset: {bias_offset}\n")

                    scale_offset = CallNodeMeta.cal_scale_offset(callnode, oc_groups)
                    callnode_meta.scale_offset.append(scale_offset)
                    self.debug_stream.write(f"\t\tscale_offset: {scale_offset}\n")
                    
                    if callnode_meta.from_dram:
                        i_tile_offset = CallNodeMeta.cal_i_tile_offset(
                            callnode, i_tile_coord, self.ch_per_group
                        )
                            # NOTE: I think there might have a information to record how many groups of (ifm_c//ch_per_group)
                        callnode_meta.i_tile_offset_map[i_tile_coord].append(i_tile_offset)
                        self.debug_stream.write(f"\t\ti_tile_offset: {i_tile_offset}\n")   
                        
                    else:
                        self.debug_stream.write(f"\t\ti_tile_offset: -1\n") 
                        callnode_meta.i_tile_offset_map[i_tile_coord].append(-1)

                    if callnode_meta.to_dram:
                        o_tile_offset = CallNodeMeta.cal_o_tile_offset(
                            callnode, oc_groups, o_tile_coord, self.ch_per_group
                        )
                        callnode_meta.o_tile_offset_map[o_tile_coord].append(o_tile_offset)
                        self.debug_stream.write(f"\t\to_tile_offset: {o_tile_offset}\n")  
                    else:
                        self.debug_stream.write(f"\t\to_tile_offset: -1\n") 
                        callnode_meta.o_tile_offset_map[o_tile_coord].append(-1)

                self.debug_stream.write(f"-----next tile------\n")
        else:
            raise ValueError(f"[TILING] ERROR THE OP {callnode.op} IS NOT SUPPORTED")
        
    def create_callnode_tile_attrs_map(self):
        """
        Create a map of callnode tile attributes. Only necessary information is created.

        Parameters
        ----------
            None.

        Returns
        -------
            Return a map of callnode tile attributes.
        """

        callnode_tile_attrs_map: Dict[
            Node, List[Dict[str, Union[List, Tuple, int, bool]]]
        ] = {} # List[Dict[str, Union[List, Tuple, int, bool]] -> List of dictionary: [key: str, value: Union of List, Tuple, int, bool]

        for callnode, meta in self.callnode_meta_map.items():
            if callnode not in callnode_tile_attrs_map:
                callnode_tile_attrs_map[callnode] = []
            for i_tile_coord in meta.i_tile_channel_queue.keys():
                o_tile_coord = meta.i_coord_o_coord_map[i_tile_coord]
                for idx in range(len(meta.i_tile_channel_queue[i_tile_coord])):
                    tile_attrs = {
                        "i_tile_shape": tuple(meta.i_tile_shape_map[i_tile_coord])
                        + (meta.i_tile_channel_queue[i_tile_coord][idx],),
                        "i_tile_offset": meta.i_tile_offset_map[i_tile_coord][idx],
                        "o_tile_shape": meta.o_tile_shape_map[o_tile_coord]
                        + (meta.o_tile_channel_queue[o_tile_coord][idx],),
                        "o_tile_offset": meta.o_tile_offset_map[o_tile_coord][idx],
                        "padding": meta.hw_coord_padding_map[i_tile_coord],
                        "weight_offset": meta.weight_offset[idx],
                        "bias_offset": meta.bias_offset[idx],
                        "scale_offset": meta.scale_offset[idx],
                        "from_dram": meta.from_dram,
                        "to_dram": meta.to_dram,
                        "o_tile_coord": o_tile_coord, # Debug usage
                        "i_tile_coord": i_tile_coord, # Debug usage
                        "oc_group": idx # Debug usage
                    }
                    callnode_tile_attrs_map[callnode].append(tile_attrs)
        return callnode_tile_attrs_map

    def debug(self, callnode: Node):
        # Create the directory if it doesn't exist
        if not os.path.exists(TILING_DEBUG_DIR):
            makedirs(TILING_DEBUG_DIR)

        # Write the log file inside tiling/testing/
        log_path = os.path.join(TILING_DEBUG_DIR, f"{callnode.name}.log")
        with open(log_path, "w") as f:
            f.write(self.debug_stream.getvalue())

        self.debug_stream.close()

    def dump(self):
        dump_stream = StringIO()
        node_name = ""

        for callnode, meta in self.callnode_meta_map.items():
            dump_stream.write(f"callnode: {callnode.op}")

            if hasattr(callnode, "stride"):
                dump_stream.write(f" stride: {callnode.stride.x}, {callnode.stride.y}\n")

            for i_tile_coord in meta.i_tile_channel_queue.keys():
                o_tile_coord = meta.i_coord_o_coord_map[i_tile_coord]
                for idx in range(len(meta.i_tile_channel_queue[i_tile_coord])):
                    dump_stream.write(f"\toc group idx: {idx}\n")
                    dump_stream.write(f"\t\ti_tile_coord(h,w): {i_tile_coord}\n")
                    dump_stream.write(
                        f"\t\ti_tile_shape: {tuple(meta.i_tile_shape_map[i_tile_coord]) + (meta.i_tile_channel_queue[i_tile_coord][idx], )}\n"
                    )
                    dump_stream.write(f"\t\ti_tile_offset: {meta.i_tile_offset_map[i_tile_coord][idx]}\n")
                    dump_stream.write(f"\t\to_tile_coord(h,w): {o_tile_coord}\n")
                    dump_stream.write(
                        f"\t\to_tile_shape: {meta.o_tile_shape_map[o_tile_coord] + (meta.o_tile_channel_queue[o_tile_coord][idx], )}\n"
                    )
                    dump_stream.write(
                        f"\t\to_tile_offset: {meta.o_tile_offset_map[o_tile_coord][idx]}\n"
                    )
                    dump_stream.write(f"\t\tpadding: {meta.hw_coord_padding_map[i_tile_coord]}\n")
                    dump_stream.write(f"\t\tweight_offset: {meta.weight_offset[idx]}\n")
                    dump_stream.write(f"\t\tbias_offset: {meta.bias_offset[idx]}\n")
                    dump_stream.write(f"\t\tscale_offset: {meta.scale_offset[idx]}\n")
                    dump_stream.write("\n")
                    node_name += "_" if node_name != "" else ""

            if hasattr(callnode, "name") and isinstance(callnode.name, str):
                node_name += f"{callnode.name}"
            else:
                node_name += f"{callnode.op}"

        if not os.path.exists(TILING_DIR):
            os.makedirs(TILING_DIR)


        # Write the log file inside build/tilings
        log_path = os.path.join(TILING_DIR, f"{node_name}.log")
        with open(log_path, "w") as f:
            f.write(dump_stream.getvalue())

        dump_stream.close()

    def __call__(self, callnode: Node):
        self.reset()
        root_callnode = callnode
        self.init_callnode_meta(root_callnode)
        self.infer_overlap(root_callnode)  # each op map to a overlap: Shape
        max_output_tile_shape = self.infer_max_out_tile_shape(root_callnode)
        # print("max_output_tile_shape", max_output_tile_shape)
        self.set_from_dram_flag(root_callnode)
        self.set_to_dram_flag(root_callnode)
        self.hw_tiling(root_callnode, max_output_tile_shape)
        self.depth_tiling(root_callnode)
        # self.debug(root_callnode)
        self.dump()
        return self.create_callnode_tile_attrs_map()

        