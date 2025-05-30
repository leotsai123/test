from typing import Dict,Union
from aiohttp import RequestInfo
import numpy as np
from macro_gen.isa_def import ConvMacroOp, TLBR, PositionMap, MemType, Dtype
from model_construct.node import HWC, HW
from npu_emulator.utils import Shape, assign_torch_dtype, dtype_to_bytes
from npu_emulator.vpu import VRF
from npu_emulator.utils import pack_int8_to_int64

class FetchUnit:
    def __init__(self, vrf: VRF, ch_per_group: int, cim_oc_: int,) :
        self.vrf = vrf
        self.input_buffer=np.zeros(8, dtype=np.int64)
        self.bias_buffer=np.zeros(cim_oc_, dtype=np.int32)
        self.fetch_ifm_cnt=0
        self.fetch_psum_cnt=0
        self.fetch_bias_cnt=0
        self.ch_per_group = ch_per_group

    def fetch_input(self, request: Dict[str, Union[Shape, HWC, HW, TLBR, PositionMap, int, Dtype, bool]]):
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
        request: o_coord, ifm_shape, kernel_shape, strides, padding, posmap, icg, k_size, ifm_sram_base, ifm_dtype, flush_en

        Return
        ------
        Return the input buffer with fetched value

        """

        o_coord =       request["o_coord"]
        ifm_shape =     request["ifm_shape"]
        kernel_shape =  request["kernel_shape"]
        strides =       request["strides"]
        padding =       request["padding"]
        k_size =        request["k_size"]
        icg =           request["icg"]
        ifm_sram_base = request["ifm_sram_base"]
        ifm_dtype =     request["ifm_dtype"]

        data_byte = dtype_to_bytes(ifm_dtype)
        posmap = request["posmap"] #TODO: add this function later
        flush_en = request["flush_en"] #TODO: add this function later

        oh, ow = o_coord.h, o_coord.w
        ifm_h, ifm_w, ifm_c = ifm_shape.h, ifm_shape.w, ifm_shape.c
        kernel_h, kernel_w = kernel_shape.h, kernel_shape.w
        stride_h, stride_w = strides.h, strides.w
        pad_top, pad_left = padding.t, padding.l
        ih_start = oh * stride_h - pad_top
        iw_start = ow * stride_w - pad_left
        ic_elements = 8 if k_size == 8 else 64 # ic elements in a segment
        # coords = []
        # TODO: consider 16-bit input transfer and align mode address mapping
        buff_idx=0
        ice_bound = ifm_c//self.ch_per_group if ifm_c < ic_elements else ic_elements//self.ch_per_group
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                if k_size == 8 and kh == kernel_h-1 and kw == kernel_w-1:
                    continue
                elif k_size == 1 and not (kh == kernel_h - 1 and kw == kernel_w - 1):
                    continue
                ih = ih_start + kh
                iw = iw_start + kw
               
                if ih < 0 or ih >= ifm_h or iw < 0 or iw >= ifm_w:
                    for ice in range(ice_bound):
                        self.input_buffer[buff_idx] = np.zeros(1, dtype=np.int64)
                        buff_idx+=1
                    continue # skip fetching

                # datalayout in vrf is also NHCWB8
                for ice in range(ice_bound):
                    addr = (
                        ih * ifm_c * ifm_w +                    # ifm tile height
                        icg * ic_elements * ifm_w +             # cim ic segment
                        ice * ifm_w * self.ch_per_group +       # cim ic element/segment
                        iw * self.ch_per_group                  # ifm tile width
                    )*data_byte + ifm_sram_base                 # base (ifm channel offset)
                    data = self.vrf.mem[addr:addr+self.ch_per_group*data_byte]
                    self.input_buffer[buff_idx] = pack_int8_to_int64(data)
                    buff_idx+=1
                    self.fetch_ifm_cnt+=1
                # fill the rest of the input buffer with zeros
                if buff_idx < len(self.input_buffer):
                    self.input_buffer[buff_idx:] = np.zeros(1, dtype=np.int64)

                # coords.append((ih,iw))
        return self.input_buffer
    
    def fetch_psum(self, request: Dict[str, Union[Shape, HWC, int, bool]], cim_oc_: int,):

        o_coord = request["o_coord"]
        ofm_shape = request["ofm_shape"]
        psum_sram_base = request["psum_sram_base"]
        cim_oc = request["cim_oc"]
        sp_group = request["sp_group"]
        psum_dtype = request["psum_dtype"]
        data_byte = dtype_to_bytes(psum_dtype)

        # psum in vrf layout is NHCWB8
        psum_tmp=np.zeros(cim_oc_, dtype=np.int32) # NOTE: this variable is used for convenience, it's not a actual register!!!
        for ocg in range(cim_oc//self.ch_per_group): # oc_segments, fetch B8 elements each
            addr = (
                o_coord.h * ofm_shape.c * ofm_shape.w + 
                sp_group * cim_oc * ofm_shape.w +
                ocg * ofm_shape.w * self.ch_per_group +
                o_coord.w * self.ch_per_group
            )* data_byte + psum_sram_base  
            data_arr = [] # this contains B8 element
            for i in range(data_byte): # for each B8, we need to fetch data_byte times
                data = pack_int8_to_int64(
                    self.vrf.mem[
                        addr+i*self.ch_per_group:
                        addr+i*self.ch_per_group+self.ch_per_group]
                )
                data_arr.append(data)
                self.fetch_psum_cnt+=1
            data_arr = np.array(data_arr)
            # convert the fetched 4 64-bit data into B8 32-bit data, and update to cim_unit psum_buffer
            buff_base = sp_group*cim_oc + ocg*self.ch_per_group
            assert buff_base+self.ch_per_group <= cim_oc_, f"psum buffer access out of bound!!! buff_base={buff_base}"
            psum_tmp[buff_base:buff_base+self.ch_per_group] = data_arr.view(np.int32)
            
        return psum_tmp
    
    def fetch_bias(self, request: Dict[str, Union[Shape, HWC, int, bool]], bias_gold):
        bias_sram_base = request["bias_sram_base"]
        cim_oc = request["cim_oc"]
        sp_group = request["sp_group"]
        bias_dtype = request["bias_dtype"]
        data_byte = dtype_to_bytes(bias_dtype)

        for ocg in range(cim_oc//self.ch_per_group):
            addr = (
                sp_group * cim_oc +
                ocg * self.ch_per_group
            )* data_byte + bias_sram_base
            
            data_arr = [] # this contains B8 element
            for i in range(data_byte):
                data = pack_int8_to_int64(
                    self.vrf.mem[
                        addr+i*self.ch_per_group:
                        addr+i*self.ch_per_group+self.ch_per_group]
                )
                data_arr.append(data)
                self.fetch_bias_cnt+=1
            data_arr = np.array(data_arr)
            buff_base = sp_group*cim_oc + ocg*self.ch_per_group
            self.bias_buffer[buff_base:buff_base+self.ch_per_group] = data_arr.view(np.int32)
        return self.bias_buffer