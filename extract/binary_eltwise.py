from .op import MacroOp
from .isa_def import (
    MacroOpType,
    Dtype,
    ActivationType,
    HWC,
    BinEltwiseOpType,
    ComponentType,
    Region,
    OpCode,
    DependencyTag,
)


class BinEltwiseMacroOp(MacroOp):
    def __init__(
        self,
        bin_eltwise_op_type=BinEltwiseOpType.ADD,
        ofm_shape=HWC(0, 0, 0),
        ifm_shape=HWC(0, 0, 0),
        ifm_sram_base=0,
        ifm_bank_start=0,
        ifm_bank_span=0,
        ifm_dtype=Dtype.INT8,
        ifm1_sram_base=0,
        ifm1_bank_start=0,
        ifm1_bank_span=0,
        ofm_sram_base=0,
        ofm_bank_start=[0, 0],
        ofm_bank_span=0,
        ofm_dtype=Dtype.INT8,
        psum_dtype=Dtype.INT24,
        req_dtype=Dtype.INT8,
        act_type=ActivationType.CLIP,
        broadcast_h=False,
        broadcast_w=False,
        broadcast_c=False,
        dep_tag=None,
    ):
        super().__init__(MacroOpType.BIN_ELTWISE, False)
        self.bin_eltwise_op_type = bin_eltwise_op_type
        self.ofm_shape = ofm_shape
        self.ifm_shape = ifm_shape
        self.ifm_sram_base = ifm_sram_base
        self.ifm_bank_start = ifm_bank_start
        self.ifm_bank_span = ifm_bank_span
        self.ifm_dtype = ifm_dtype
        self.ifm1_sram_base = ifm1_sram_base
        self.ifm1_bank_start = ifm1_bank_start
        self.ifm1_bank_span = ifm1_bank_span
        self.ofm_sram_base = ofm_sram_base
        self.ofm_bank_start = ofm_bank_start
        self.ofm_bank_span = ofm_bank_span
        self.ofm_dtype = ofm_dtype
        self.psum_dtype = psum_dtype
        self.req_dtype = req_dtype
        self.act_type = act_type
        self.broadcast_h = broadcast_h
        self.broadcast_w = broadcast_w
        self.broadcast_c = broadcast_c
        self.dep_tag = dep_tag

    def emit(self, emitter):
        emitter.set_target(ComponentType.PPU)
        ppu_config = emitter.ppu_config_reg
        ifm_config = {
            OpCode.IFM_HEIGHT: self.ifm_shape.h,
            OpCode.IFM_WIDTH: self.ifm_shape.w,
            OpCode.IFM_CHANNEL: self.ifm_shape.c,
            OpCode.IFM_BASE: self.ifm_sram_base,
            OpCode.IFM2_BASE: self.ifm1_sram_base,
            OpCode.IFM_REGION: [
                Region.O_SRAM,
                [self.ifm_bank_start, self.ifm_bank_start + self.ifm_bank_span],
            ],
            OpCode.IFM2_REGION: [
                Region.O_SRAM,
                [self.ifm1_bank_start, self.ifm1_bank_start + self.ifm1_bank_span],
            ],
        }
        ppu_config.ifm_config_reg.emit(emitter, ifm_config)

        ofm_config = {
            OpCode.OFM_HEIGHT: self.ofm_shape.h,
            OpCode.OFM_WIDTH: self.ofm_shape.w,
            OpCode.OFM_CHANNEL: self.ofm_shape.c,
            OpCode.OFM_BASE: self.ofm_sram_base,
            OpCode.OFM_REGION: [
                Region.O_SRAM,
                [self.ofm_bank_start, self.ofm_bank_start + self.ofm_bank_span],
            ],
        }
        ppu_config.ofm_config_reg.emit(emitter, ofm_config)

        quantization_config = {
            OpCode.ACTIVATION_FUNCTION_TYPE: self.act_type,
            OpCode.CLIP_MIN: 0,
            OpCode.CLIP_MAX: 255,
        }
        ppu_config.quantization_config_reg.emit(emitter, quantization_config)

        misc_config = {
            OpCode.DATA_TYPE: [
                self.ifm_dtype,
                None,
                self.ofm_dtype,
                self.psum_dtype,
                self.req_dtype,
            ],
        }
        ppu_config.misc_config_reg.emit(emitter, misc_config)

        emitter.emit_bin_eltwise(
            self.bin_eltwise_op_type,
            self.broadcast_h,
            self.broadcast_w,
            self.broadcast_c,
            self.dep_tag,
        )
