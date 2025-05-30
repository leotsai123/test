from .op import MacroOp
from .isa_def import (
    MacroOpType,
    Dtype,
    ActivationType,
    ScaleModeType,
    HWC,
    UnEltwiseOpType,
    ComponentType,
    Region,
    OpCode,
    DependencyTag,
)


class UnEltwiseMacroOp(MacroOp):
    def __init__(
        self,
        un_eltwise_op_type=UnEltwiseOpType.CLZ,
        ofm_shape=HWC(0, 0, 0),
        ifm_shape=HWC(0, 0, 0),
        ifm_sram_base=0,
        ifm_bank_start=0,
        ifm_bank_span=0,
        ifm_dtype=Dtype.INT8,
        ifm_zp=0,
        scale_mantissa=0,
        scale_shift=0,
        scale_base=0,
        ofm_sram_base=0,
        ofm_bank_start=[0, 0],
        ofm_bank_span=0,
        ofm_dtype=Dtype.INT8,
        ofm_zp=0,
        psum_dtype=Dtype.INT24,
        req_dtype=Dtype.INT8,
        act_type=ActivationType.CLIP,
        scale_mode=ScaleModeType.PER_TENSOR_AFFINE,
        req_en=False,
        imm_low=None,
        imm_high=None,
        dep_tag=None,
    ):
        super().__init__(MacroOpType.UN_ELTWISE, False)
        self.un_eltwise_op_type = un_eltwise_op_type
        self.ofm_shape = ofm_shape
        self.ifm_shape = ifm_shape
        self.ifm_sram_base = ifm_sram_base
        self.ifm_bank_start = ifm_bank_start
        self.ifm_bank_span = ifm_bank_span
        self.ifm_dtype = ifm_dtype
        self.ifm_zp = ifm_zp
        self.scale_mantissa = scale_mantissa
        self.scale_shift = scale_shift
        self.scale_base = scale_base
        self.ofm_sram_base = ofm_sram_base
        self.ofm_bank_start = ofm_bank_start
        self.ofm_bank_span = ofm_bank_span
        self.ofm_dtype = ofm_dtype
        self.ofm_zp = ofm_zp
        self.psum_dtype = psum_dtype
        self.req_dtype = req_dtype
        self.act_type = act_type
        self.scale_mode = scale_mode
        self.req_en = req_en
        self.imm_low = imm_low
        self.imm_high = imm_high
        self.dep_tag = dep_tag

    def emit(self, emitter):
        emitter.set_target(ComponentType.PPU)
        ppu_config = emitter.ppu_config_reg
        ifm_config = {
            OpCode.IFM_HEIGHT: self.ifm_shape.h,
            OpCode.IFM_WIDTH: self.ifm_shape.w,
            OpCode.IFM_CHANNEL: self.ifm_shape.c,
            OpCode.IFM_ZERO_POINT: self.ifm_zp,
            OpCode.IFM_BASE: self.ifm_sram_base,
            OpCode.IFM_REGION: [
                Region.O_SRAM,
                [self.ifm_bank_start, self.ifm_bank_start + self.ifm_bank_span],
            ],
        }
        ppu_config.ifm_config_reg.emit(emitter, ifm_config)

        ofm_config = {
            OpCode.OFM_HEIGHT: self.ofm_shape.h,
            OpCode.OFM_WIDTH: self.ofm_shape.w,
            OpCode.OFM_CHANNEL: self.ofm_shape.c,
            OpCode.OFM_ZERO_POINT: self.ofm_zp,
            OpCode.OFM_BASE: self.ofm_sram_base,
            OpCode.OFM_REGION: [
                Region.O_SRAM,
                [self.ofm_bank_start, self.ofm_bank_start + self.ofm_bank_span],
            ],
        }
        ppu_config.ofm_config_reg.emit(emitter, ofm_config)

        quantization_config = {
            OpCode.QUANTIZATION_MODE: [self.scale_mode, self.ifm_zp != 0, False, self.ofm_zp != 0],
            OpCode.ACTIVATION_FUNCTION_TYPE: self.act_type,
            OpCode.CLIP_MIN: 0,
            OpCode.CLIP_MAX: 255,
            OpCode.SCALE_MANTISSA: self.scale_mantissa,
            OpCode.SCALE_SHIFT: self.scale_shift,
            OpCode.SCALE_BASE: self.scale_base,
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
            OpCode.IMM_LOW: self.imm_low,
            OpCode.IMM_HIGH: self.imm_high,
        }
        ppu_config.misc_config_reg.emit(emitter, misc_config)

        emitter.emit_un_eltwise(self.un_eltwise_op_type, self.dep_tag)
