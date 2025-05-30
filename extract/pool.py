from .op import MacroOp
from .window import WindowInfo
from .isa_def import (
    MacroOpType,
    Dtype,
    ActivationType,
    ScaleModeType,
    HWC,
    PoolOpType,
    ComponentType,
    Region,
    OpCode,
    DependencyTag,
)


class PoolMacroOp(MacroOp):
    def __init__(
        self,
        ofm_shape=HWC(0, 0, 0),
        ifm_shape=HWC(0, 0, 0),
        ifm_sram_base=0,
        ifm_bank_start=0,
        ifm_bank_span=0,
        ifm_dtype=Dtype.INT8,
        ifm_zp=0,
        window_info=WindowInfo(),
        bias_base=0,
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
        pool_op_type=PoolOpType.AVG_POOL,
        accu_en=False,
        req_en=False,
        dep_tag=None,
    ):
        super().__init__(MacroOpType.POOL, True)
        self.ofm_shape = ofm_shape
        self.ifm_shape = ifm_shape
        self.ifm_sram_base = ifm_sram_base
        self.ifm_bank_start = ifm_bank_start
        self.ifm_bank_span = ifm_bank_span
        self.ifm_dtype = ifm_dtype
        self.ifm_zp = ifm_zp
        self.window_info = window_info
        self.bias_base = bias_base
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
        self.pool_op_type = pool_op_type
        self.accu_en = accu_en
        self.req_en = req_en
        self.dep_tag = dep_tag

    def emit(self, emitter):
        emitter.set_target(ComponentType.PPU)
        ppu_config = emitter.ppu_config_reg
        ifm_config = {
            OpCode.IFM_HEIGHT: self.ifm_shape.h,
            OpCode.IFM_WIDTH: self.ifm_shape.w,
            OpCode.IFM_CHANNEL: self.ifm_shape.c,
            OpCode.IFM_PAD: self.window_info.padding,
            OpCode.IFM_UPSAMPLING_RATIO: self.window_info.upsample_ratio,
            OpCode.IFM_ZERO_POINT: self.ifm_zp,
            OpCode.IFM_BASE: self.ifm_sram_base,
            OpCode.IFM_REGION: [
                Region.O_SRAM,
                [self.ifm_bank_start, self.ifm_bank_start + self.ifm_bank_span],
            ],
        }
        ppu_config.ifm_config_reg.emit(emitter, ifm_config)

        weight_config = {
            OpCode.KERNEL_SHAPE: self.window_info.kernel_shape,
            OpCode.KERNEL_STRIDE: self.window_info.strides,
            OpCode.KERNEL_DILATION: self.window_info.dilation,
        }
        ppu_config.weight_config_reg.emit(emitter, weight_config)

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
            OpCode.QUANTIZATION_MODE: [
                self.scale_mode,
                self.ifm_zp != 0,
                False,
                self.ofm_zp != 0,
            ],
            OpCode.ACTIVATION_FUNCTION_TYPE: self.act_type,
            OpCode.CLIP_MIN: 0,
            OpCode.CLIP_MAX: 255,
            OpCode.SCALE_MANTISSA: self.scale_mantissa,
            OpCode.SCALE_SHIFT: self.scale_shift,
            OpCode.BIAS_BASE: self.bias_base,
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
        }
        ppu_config.misc_config_reg.emit(emitter, misc_config)

        emitter.emit_pool(self.pool_op_type, self.accu_en, self.req_en, self.dep_tag)
