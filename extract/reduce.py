from .op import MacroOp
from .isa_def import (
    MacroOpType,
    Dtype,
    ActivationType,
    ScaleModeType,
    HWC,
    ComponentType,
    ReduceOpType,
    DependencyTag,
)


class ReduceMacroOp(MacroOp):
    def __init__(
        self,
        reduce_op_type=ReduceOpType.SUM,
        reduce_dim=HWC(False, False, False),
        ofm_shape=HWC(0, 0, 0),
        ifm_shape=HWC(0, 0, 0),
        ifm_sram_base=0,
        ifm_bank_start=0,
        ifm_bank_span=0,
        ifm_dtype=Dtype.INT8,
        scale_mantissa=0,
        scale_shift=0,
        scale_base=0,
        ofm_sram_base=0,
        ofm_bank_start=[0, 0],
        ofm_bank_span=0,
        ofm_dtype=Dtype.INT8,
        psum_dtype=Dtype.INT24,
        req_dtype=Dtype.INT8,
        act_type=ActivationType.CLIP,
        scale_mode=ScaleModeType.PER_TENSOR_AFFINE,
        req_en=False,
        dep_tag=DependencyTag(0, 0, 0, 0),
    ):
        super().__init__(MacroOpType.REDUCE, False)
        assert not reduce_dim.c
        self.reduce_op_type = reduce_op_type
        self.reduce_dim = reduce_dim
        self.ofm_shape = ofm_shape
        self.ifm_shape = ifm_shape
        self.ifm_sram_base = ifm_sram_base
        self.ifm_bank_start = ifm_bank_start
        self.ifm_bank_span = ifm_bank_span
        self.ifm_dtype = ifm_dtype
        self.scale_mantissa = scale_mantissa
        self.scale_shift = scale_shift
        self.scale_base = scale_base
        self.ofm_sram_base = ofm_sram_base
        self.ofm_bank_start = ofm_bank_start
        self.ofm_bank_span = ofm_bank_span
        self.ofm_dtype = ofm_dtype
        self.psum_dtype = psum_dtype
        self.req_dtype = req_dtype
        self.act_type = act_type
        self.scale_mode = scale_mode
        self.req_en = req_en
        self.dep_tag = dep_tag

    def emit(self, emitter):
        emitter.set_target(ComponentType.PPU)
