from typing import List, Tuple, Dict, Union
from macro_gen.isa_def import ConvMacroOp, LoadMacroOp, LoadMicroOp_, LoadCimMacroOp, MemType, SanityCheckOp, StoreMacroOp
from model_construct.layout import to_nhcwb8, to_ochwb8
from model_construct.node import Node
import numpy as np
import torch

from npu_emulator.vpu import VRF, VPU
from npu_emulator.top_controller import TopController
from npu_emulator.fetch_uinit import FetchUnit
from npu_emulator.cim_unit import CimUnit
from npu_emulator.store_unit import StoreUnit
from npu_emulator.validator import EmulateValidator


DRAM_SIZE = 1024 * 1024 * 10 # 10MB

class EmulateBuilder:
    def __init__(
        self, 
        callnode: Node, 
        macro_ops: List[Union[ConvMacroOp, LoadMacroOp, LoadCimMacroOp, StoreMacroOp]], 
        ch_per_group: int, 
        slide_size: int, # hardware contraint
        cim_oc_: int, # hardware contraint
        validator: EmulateValidator,
        layer_idx: int,
        mode: str
    ):
        self.callnode = callnode
        self.mode = mode
        self.ifm_extern = validator.ifm_extern
        self.weight_extern = validator.weight_extern
        self.bias_extern = validator.bias_extern
        self.ofm_extern = np.zeros(DRAM_SIZE, dtype=np.int8)
        self.validator = validator
        self.macro_ops = macro_ops
        self.ch_per_group = ch_per_group
        self.vrf = VRF()
        self.cim_unit = CimUnit(slide_size=slide_size, cim_oc_=cim_oc_)
        self.fetch_unit = FetchUnit(self.vrf, ch_per_group=ch_per_group, cim_oc_=cim_oc_,)
        self.store_unit = StoreUnit(self.vrf, ch_per_group=ch_per_group, cim_oc_=cim_oc_)
        self.npu = TopController(self.fetch_unit, self.cim_unit, self.store_unit, ch_per_group)
        self.vpu = VPU(self.vrf, self.cim_unit.cim_macro)
        

    def __call__(self):
        for macro_op in self.macro_ops:
            if isinstance(macro_op, ConvMacroOp):
                self.validator.mac_op(
                    ofm_shape=macro_op.ofm_shape,
                    ifm_shape=macro_op.ifm_shape,
                    ifm_sram_base=macro_op.ifm_sram_base,
                    window_info=macro_op.window_info,
                    bias_sram_base=macro_op.bias_sram_base,
                    ofm_sram_base=macro_op.ofm_sram_base,
                    psum_sram_base=macro_op.psum_sram_base,
                    accu_en=macro_op.accu_en,
                    req_en=macro_op.req_en,
                    bias_en=macro_op.bias_en,
                    scale_mantissa=macro_op.scale_mantissa,
                    scale_shift=macro_op.scale_shift,
                    i_tile_coord=macro_op.i_tile_coord,
                    oc_group=macro_op.oc_group,
                    sp_group=macro_op.sp_group,
                    o_tile_coord=macro_op.o_tile_coord,
                    ping=macro_op.ping,
                    k_size=macro_op.k_size,
                    cim_ic=macro_op.cim_ic,
                    cim_oc=macro_op.cim_oc,
                    cim_sic=macro_op.cim_sic,
                    cim_soc=macro_op.cim_soc,
                    ifm_dtype=macro_op.ifm_dtype,
                    weight_dtype=macro_op.kernel_dtype,
                    psum_dtype=macro_op.psum_dtype,
                    ofm_dtype=macro_op.ofm_dtype,
                    act_type=macro_op.act_type,
                    act_max=macro_op.act_max,
                    act_min=macro_op.act_min,
                    align_mode=macro_op.align_mode,
                    overwrite=macro_op.overwrite,
                )
                if self.mode == "low":
                    self.npu.execute(
                        params=macro_op,
                        validator=self.validator
                    )
            if isinstance(macro_op, LoadMacroOp):
                self.validator.load_tile(
                    src=macro_op.src,
                    src_len=macro_op.src_len,
                    dst=macro_op.dst,
                    dst_type=macro_op.dst_region_type,
                    src_type=macro_op.src_region_type
                )
            if isinstance(macro_op, LoadMicroOp_):
                if macro_op.dst_type == MemType.UNI_SRAM and macro_op.src_type == MemType.DRAM:
                    self.vpu.load_vrf(
                        src_type    =self.ifm_extern,
                        seg_num     = macro_op.seg_num,
                        v_stride    = macro_op.v_stride,
                        seg_stride  = macro_op.seg_stride,
                        seg_len     = macro_op.seg_len,
                        src_base_addr = macro_op.src_base_addr,
                        dst_base_addr   = macro_op.dst_base_addr,
                    )
                if macro_op.dst_type == MemType.UNI_SRAM and macro_op.src_type == MemType.BIAS_MEM:
                    self.vpu.load_vrf(
                        src_type    =self.bias_extern,
                        seg_num     = macro_op.seg_num,
                        v_stride    = macro_op.v_stride,
                        seg_stride  = macro_op.seg_stride,
                        seg_len     = macro_op.seg_len,
                        src_base_addr = macro_op.src_base_addr,
                        dst_base_addr   = macro_op.dst_base_addr,
                    )
            if isinstance(macro_op, LoadCimMacroOp):
                    self.vpu.load_weight(
                        soc=macro_op.soc,
                        oc_stride=macro_op.oc_stride,
                        sic=macro_op.sic,
                        ic_stride=macro_op.ic_stride,
                        eic=macro_op.eic,
                        k_stride=macro_op.k_stride,
                        src_base_addr=macro_op.src_base_addr,
                        src_len=macro_op.src_len,
                        ping=macro_op.ping,
                        k_size=macro_op.k_size,
                        src_type=self.weight_extern,
                        kernel_hw=macro_op.kernel_hw,
                    )
            if isinstance(macro_op, StoreMacroOp):
                if macro_op.dst_region_type == MemType.DRAM:
                    self.validator.store_tile(
                        src=macro_op.src,
                        src_len=macro_op.src_len,
                        dst=macro_op.dst,
                        dst_type=macro_op.dst_region_type
                    )
                    if self.mode == "low":
                        self.vpu.store_ofm(
                                dst_type=self.ofm_extern,
                                src=macro_op.src,
                                len=macro_op.src_len,
                                dst=macro_op.dst,
                            )
            if isinstance(macro_op, SanityCheckOp):
                self.validator.sanity_check(tile_attrs=macro_op.tile_attrs)
                if self.mode == "low":
                    self.validator.tile_result_check(tile_attrs=macro_op.tile_attrs, ofm_extern=self.ofm_extern)
                    ofm_extern=self.ofm_extern