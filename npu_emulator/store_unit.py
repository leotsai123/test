import numpy as np
from typing import List, Dict, Union
from macro_gen.isa_def import ScaleModeType, ActivationType
from npu_emulator.utils import Shape, assign_np_dtype
from model_construct.node import HWC, HW
from macro_gen.isa_def import Dtype, MemType
from npu_emulator.utils import Shape, assign_torch_dtype, dtype_to_bytes
from npu_emulator.utils import pack_int8_to_int64
from npu_emulator.vpu import VRF



class StoreUnit:
    def __init__(self, vrf: VRF, cim_oc_: int, ch_per_group: int):
        self.ch_per_group = ch_per_group
        self.vrf = vrf
        self.result_buffer = np.zeros(cim_oc_, dtype=np.int32)

    def get_result(self, psum_buffer: np.array, cim_oc: int, sp_group: int, scale_mode: ScaleModeType, scale_mantissa: int, scale_shift: int, act_type: ActivationType, act_max: int, act_min: int, ofm_dtype):
        """
        Parameters
        ----------
            psum_buffer : cim_oc x 32bit regsters, cim_oc can either be 16 or 32 for now
            scale_mode: only support PER_TENSOR_AFFINE quantization
            act_type: only support RELU, CLAMP for now, they all basically clip the value within act_min and act_max
        """
        result_buffer = self.result_buffer[sp_group*cim_oc:sp_group*cim_oc+cim_oc]
        if scale_mode == ScaleModeType.PER_TENSOR_AFFINE:
            result_buffer = (psum_buffer*int(scale_mantissa)) >> int(scale_shift)
        else: raise ValueError(f"Only PER_TENSOR_AFFINE scale mode is supported, but got {scale_mode}!")

        if act_type == ActivationType.RELU or act_type == ActivationType.CLAMP:
            result_buffer = np.clip(result_buffer, act_min, act_max)
        else: raise ValueError(f"Only RELU and CLAMP activation type are supported, but got {act_type}")

        return result_buffer.astype(assign_np_dtype(ofm_dtype))

    def store_result(self, request: Dict[str, Union[Shape, HWC, HW, int, Dtype, bool]]):
        psum_buffer     = request["psum_buffer"]
        o_coord         = request["o_coord"]
        ofm_shape       = request["ofm_shape"]
        cim_oc          = request["cim_oc"]
        sp_group        = request["sp_group"]
        req_en          = request["req_en"]
        scale_mode      = request["scale_mode"]
        scale_mantissa  = request["scale_mantissa"]
        scale_shift     = request["scale_shift"]
        act_type        = request["act_type"]
        act_max         = request["act_max"]
        act_min         = request["act_min"]
        ofm_dtype       = request["ofm_dtype"]
        psum_dtype      = request["psum_dtype"]
        ofm_sram_base   = request["ofm_sram_base"]
        psum_sram_base  = request["psum_sram_base"]
        overwrite       = request["overwrite"]
        align_mode      = request["align_mode"]
        
        # result is of shape cim_oc of ofm_dtype if req_en, else psum_dtype
        result = self.get_result(
            psum_buffer[sp_group*cim_oc:sp_group*cim_oc+cim_oc],
            cim_oc,
            sp_group,
            scale_mode,
            scale_mantissa,
            scale_shift,
            act_type,
            act_max,
            act_min,
            ofm_dtype
        ) if req_en else psum_buffer[sp_group*cim_oc:sp_group*cim_oc+cim_oc]
        
        if not req_en:
            data_bytes = dtype_to_bytes(psum_dtype)
        else:
            data_bytes = dtype_to_bytes(psum_dtype) if overwrite else dtype_to_bytes(ofm_dtype)

        cim_col = self.ch_per_group
        sram_base = ofm_sram_base if req_en else psum_sram_base
        for ocg in range(cim_oc//cim_col):# oc_segments
            addr = (
                o_coord.h * ofm_shape.c * ofm_shape.w + 
                sp_group * cim_oc * ofm_shape.w +
                ocg * ofm_shape.w * cim_col +
                o_coord.w * cim_col
            )* data_bytes + sram_base  
            
            buff_base=ocg*cim_col
            data = result[buff_base:buff_base+cim_col] # this contains B8 elements
            data = data.reshape(-1).view(np.int8) # this contains flattened B8 element in byte array
            for i in range(len(data)//self.ch_per_group): # store 64bits data at a time
                base=addr+i*self.ch_per_group
                self.vrf.mem[base:base+self.ch_per_group] = data[i*self.ch_per_group:(i+1)*self.ch_per_group]

        

