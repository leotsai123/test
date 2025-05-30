from macro_gen.isa_def import ConvMacroOp, TLBR, PositionMap, MemType
from model_construct.node import HWC, HW
from tiling.tiling import Shape

import torch
from typing import List, Tuple, Dict
import numpy as np
import math


class TopController:
    def __init__(self, params: ConvMacroOp, ch_per_group): # params include all CONV macro_op fields
        self.params = params
        self.ch_per_group = ch_per_group
    
    def execute(self, fetch_unit, cim_unit, store_unit):
        ofm_h, ofm_w = self.params.ofm_shape.h, self.params.ofm_shape.w
        ifm_h, ifm_w, ifm_c = self.params.ifm_shape.h, self.params.ifm_shape.w, self.params.ifm_shape.c
        p_ifm_c = math.ceil(ifm_c/self.ch_per_group) * self.ch_per_group
        cim_ic, cim_oc = self.params.cim_ic, self.params.cim_oc
        window_info = self.params.window_info
        kernel_shape = window_info.kernel_shape
        strides = window_info.strides
        padding = window_info.padding
        posmap = self.params.posmap

        ofm_h_ = (ifm_h + padding.t + padding.b - (kernel_shape.h - 1) - 1) // strides.h + 1
        assert ofm_h == ofm_h_, "calculated ofm tile height is out of bound"
        ofm_w_ = (ifm_w + padding.l + padding.r - (kernel_shape.w - 1) - 1) // strides.w + 1
        assert ofm_w == ofm_w_, "calculated ofm tile width is out of bound"

        for oh in range(ofm_h):
            for ow in range(ofm_w):
                for ic in range(cim_ic // self.ch_per_group): # ic segments
                    fetch_request = {
                        "o_coord": Shape(oh,ow),
                        "ifm_shape": HWC(ifm_h, ifm_w, p_ifm_c), # NOTE: ifm_c is padded to align B8
                        "kernel_shape": kernel_shape,
                        "strides": strides,
                        "padding": padding,
                        "posmap": posmap,
                        "icg": ic,
                        "k_size": self.params.k_size,
                        "ifm_sram_base": self.params.ifm_sram_base,
                        "cim_ic": cim_ic
                    }
                    input_buffer = fetch_unit.fetch_input(fetch_request)
                    psum_buffer = fetch_unit.fetch_psum(fetch_request)
                    conv_result = cim_unit.compute(input_buffer, psum_buffer)

class FetchUnit:
    def __init__(self, vrf: VRF, ifm_dtype: str, ch_per_group: int):
        self.vrf = vrf
        self.input_buffer=np.zeros(8, dtype=np.int64)
        self.ifm_dtype = ifm_dtype
        self.fetch_cnt=0
        self.ch_per_group = ch_per_group

    @staticmethod
    def pack_int8_to_int64(values):
        assert len(values) == 8, "fetched input isn't B8 aligned!!!"
        packed = np.int64(0)
        for i, val in enumerate(values):
            packed |= np.int64(val) << (8 * i)
        return packed

    def fetch_input(self, request: Dict[Shape, HWC, HW, HW, TLBR, PositionMap, int, int, int, int]):
        """
        Description
        -----------
            top left corner of the required patch
            -------
            ih_start = oh * stride_h - pad_top
            iw_start = ow * stride_w - pad_left

            input spans
            -----------
            ih in range(ih_start, ih_start+k_h-1)
            iw in range(iw_start, iw_start+k_w-1)

        Parameters
        ----------
        request: o_coord, ifm_shape, kernel_shape, strides, padding, posmap, icg, k_size, ifm_sram_base, cim_ic

        Return
        ------
        Return the input buffer with fetched value

        """

        o_coord = request["o_coord"]
        ifm_shape = request["ifm_shape"]
        kernel_shape = request["kernel_shape"]
        strides = request["strides"]
        padding = request["padding"]
        k_size = request["k_size"]
        icg = request["icg"]
        cim_ic = request["cim_ic"]
        ifm_sram_base = request["ifm_sram_base"]
        posmap = request["posmap"] #TODO: add this function later

        oh, ow = o_coord.h, o_coord.w
        ifm_h, ifm_w, ifm_c = ifm_shape.h, ifm_shape.w, ifm_shape.c
        kernel_h, kernel_w = kernel_shape.h, kernel_shape.w
        stride_h, stride_w = strides.h, strides.w
        pad_top, pad_left = padding.t, padding.l
        ih_start = oh * stride_h - pad_top
        iw_start = ow * stride_w - pad_left
        coords = []

        for kh in range(kernel_h):
            for kw in range(kernel_w):
                buff_idx=0
                if k_size == 8 and kw == kernel_w - 1:
                    continue
                elif k_size == 1 and not (kh == kernel_h - 1 and kw == kernel_w - 1):
                    continue
                ih = ih_start + kh
                iw = iw_start + kw
               
                if ih < 0 or ih >= ifm_h or iw < 0 or iw >= ifm_w:
                    self.input_buffer[buff_idx] = np.zeros(1, dtype=np.int64)
                    buff_idx+=1
                    continue # skip fetching

                # datalayout in vrf is also NHCWB8
                addr = (
                    ih * ifm_c * ifm_w +        # ifm tile height
                    icg * cim_ic * ifm_w +      # ifm channel group, grouped as cim_ic
                    iw * self.ch_per_group +    # ifm tile weight
                    ifm_sram_base               # base (ifm channel group)
                )
                self.input_buffer[buff_idx] = FetchUnit.pack_int8_to_int64(self.vrf.mem[addr:addr+self.ch_per_group])
                buff_idx+=1
                self.fetch_cnt+=1
                coords.append((ih,iw))
        return self.input_buffer

class VRF:
    def __init__(self, size: int = 16 * 1024):  # 16KB
        # view vrf as a 1D array with byte address
        self.mem = np.zeros(size, dtype=np.int8)

    def read(self, addr: int, length: int) -> np.array:
        return self.mem[addr:addr+length].clone()

    def write(self, addr: int, data: np.array):
        self.mem[addr:addr+len(data)] = data

class CIM:
    def __init__(self, size: int = 64*8*8):  # 4KB
        self.mem = np.zeros(size, dtype=np.int8)

class VPU:
    def __init__(self, vrf: VRF):
        self.vrf = vrf

    def load_ifm_tile(self, ifm_data: np.ndarray, base_addr: int):
        self.vrf.write(base_addr, ifm_data)

    def load_weight_tile(self, weight_data: np.ndarray, base_addr: int):
        self.vrf.write(base_addr, weight_data)

