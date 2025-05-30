from enum import Enum, auto
from model_construct.node import HWC, HW
from macro_gen.op import MacroOp, MicroOp
from tiling.tiling import Shape
import numpy as np 

class MemType(Enum):
    DRAM = 0
    UNI_SRAM = 1
    W_SRAM = 2
    BIAS_MEM = 3 # temporary, TODO: should be integrated into DRAM with ifm/weight/bias

class CommonConfig:
    def __str__(self):
        var_list = {
            k: v for k, v in vars(self).items() if not k.startswith("_") and not callable(v)
        }
        return "{}({})".format(self.__class__.__name__, var_list)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return vars(self) == vars(other)
    
class Dtype(Enum):
    INT8 = 0
    UINT8 = 1
    INT16 = 2
    UINT16 = 3
    INT32 = 4
    UINT32 = 5

class TLBR(CommonConfig):
    def __init__(self, t, l, b, r):
        self.t = t
        self.l = l
        self.b = b
        self.r = r

class ActivationType(Enum):
    NONE = 0
    CLAMP = 1
    RELU = 2

class PositionMap(CommonConfig):
    def __init__(
        self,
        p1=0,
        p2=0,
        p3=0,
        p4=0,
        p5=0,
        p6=0,
        p7=0,
        p8=0,
    ):
        value = (
            p1 << 0 |
            p2 << 1 |
            p3 << 2 |
            p4 << 3 |
            p5 << 4 |
            p6 << 5 |
            p7 << 6 |
            p8 << 7
        )
        self.posmap = np.uint8(value)
        


class WindowInfo(CommonConfig):
    def __init__(
        self,
        padding=TLBR(0, 0, 0, 0),
        upsample_ratio=HW(1, 1),
        kernel_shape=HW(1, 1),
        strides=HW(1, 1),
        dilation=HW(1, 1),
    ):
        self.padding = padding
        self.upsample_ratio = upsample_ratio
        self.kernel_shape = kernel_shape
        self.strides = strides
        self.dilation = dilation

class ScaleModeType(Enum):
    PER_TENSOR_POT = 0x00       # per tensor power of two -> shift only
    PER_TENSOR_AFFINE = 0x10    # per tensor affine -> mantissa and shift
    PER_CHANNEL_POT = 0x01      # per channel power of two -> shift only   
    PER_CHANNEL_AFFINE = 0x11   # per channel affine -> mantissa and shift


class MacroOpType(Enum):
    CONV = 0x0
    LOAD = 0x1
    STORE = 0x2

class MicroOpType(Enum):
    LOAD = 0x0
    STORE = 0x1


class XY(CommonConfig):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class XYZ(XY):
    def __init__(self, x, y, z):
        super().__init__(x, y)
        self.z = z

class LoadMacroOp_(MacroOp):
    def __init__(
        self,
        xyz_len: XYZ,
        src_base_addr: int,
        dst_base_addr: int,
        src_type: MemType,
        dst_type: MemType,
        src_stride: XY,
        dst_stride: XY,
        continuous_flag: bool=False,
    ):
        super().__init__(MacroOpType.LOAD, False)
        self.xyz_len=xyz_len
        self.src_base_addr=src_base_addr
        self.dst_base_addr=dst_base_addr
        self.src_type=src_type
        self.dst_type=dst_type
        self.src_stride=src_stride
        self.dst_stride=dst_stride
        self.continuous_flag=continuous_flag

class LoadMicroOp_(MicroOp):
    def __init__(
        self,
        src_type: MemType.DRAM,
        seg_num: int,
        v_stride: int,
        seg_stride: int,
        seg_len: int,
        src_base_addr: int,
        dst_base_addr: int,
        dst_type: MemType.UNI_SRAM,
    ):
        super().__init__(MicroOpType.LOAD)
        self.src_type=src_type
        self.seg_num=seg_num
        self.v_stride=v_stride
        self.seg_stride=seg_stride
        self.dst_type=dst_type
        self.seg_len=seg_len
        self.src_base_addr=src_base_addr
        self.dst_base_addr=dst_base_addr


class LoadMacroOp(MacroOp):
    def __init__(
        self,
        src_region_type: MemType.DRAM, 
        src_len: int, 
        src: int, 
        dst_region_type: MemType.UNI_SRAM,
        dst: int,
    ):
        super().__init__(MacroOpType.LOAD, False)
        self.src_region_type=src_region_type
        self.src_len=src_len
        self.src=src
        self.dst_region_type=dst_region_type
        self.dst=dst

class LoadCimMacroOp(MacroOp):
    def __init__(
        self,
        soc: int,
        oc_stride: int,
        sic: int,
        ic_stride: int,
        eic: int,
        k_stride: int,
        src_base_addr: int,
        src_len: int,
        ping: bool,
        k_size: int,
        kernel_hw: tuple,
    
    ):
        super().__init__(MacroOpType.LOAD, False)
        self.soc=soc
        self.oc_stride=oc_stride
        self.sic=sic
        self.ic_stride=ic_stride
        self.eic=eic
        self.k_stride=k_stride
        self.src_base_addr=src_base_addr
        self.src_len=src_len
        self.ping=ping
        self.k_size=k_size
        self.kernel_hw=kernel_hw

class StoreMacroOp(MacroOp):
    def __init__(
        self,
        src_region_type: MemType.UNI_SRAM, 
        src_len: int, 
        src: int, 
        dst_region_type: MemType.DRAM,
        dst: int
    ):
        super().__init__(MacroOpType.STORE, False)
        self.src_region_type=src_region_type
        self.src_len=src_len
        self.src=src
        self.dst_region_type=dst_region_type
        self.dst=dst

class ConvMacroOp(MacroOp):
    def __init__(
        self,
        ofm_shape=HWC(0, 0, 0),
        ifm_shape=HWC(0, 0, 0),
        ifm_sram_base=0,
        ifm_dtype=Dtype.INT8,
        ifm_zp=0,
        window_info=WindowInfo(),
        posmap=PositionMap(),
        kernel_dtype=Dtype.UINT8,
        bias_sram_base=0,
        scale_mantissa=0,
        scale_shift=0,
        scale_base=0,
        ofm_sram_base=0,
        psum_sram_base=0,
        ofm_dtype=Dtype.INT8,
        ofm_zp=0,
        psum_dtype=Dtype.INT32,
        req_dtype=Dtype.INT8,
        act_type=ActivationType.NONE,
        act_max=int,
        act_min=int,
        scale_mode=ScaleModeType.PER_TENSOR_AFFINE,
        accu_en=False,
        req_en=False,
        bias_en=False,
        ping=True,
        k_size=int,
        cim_sic=int,
        cim_soc=int,
        sp_group=int,
        cim_ic=int,
        cim_oc=int,
        overwrite=bool,
        align_mode=bool,
        i_tile_coord=Shape,
        oc_group=int,
        o_tile_coord=Shape,
    ):
        super().__init__(MacroOpType.CONV, True)
        self.ofm_shape = ofm_shape
        self.ifm_shape = ifm_shape
        self.ifm_sram_base = ifm_sram_base
        self.ifm_dtype = ifm_dtype
        self.ifm_zp = ifm_zp
        self.window_info = window_info
        self.posmap = posmap
        self.kernel_dtype = kernel_dtype
        self.bias_sram_base = bias_sram_base
        self.scale_mantissa = scale_mantissa
        self.scale_shift = scale_shift
        self.scale_base = scale_base
        self.ofm_sram_base = ofm_sram_base
        self.psum_sram_base=psum_sram_base
        self.ofm_dtype = ofm_dtype
        self.ofm_zp = ofm_zp
        self.psum_dtype = psum_dtype
        self.req_dtype = req_dtype
        self.act_type = act_type
        self.act_max = act_max
        self.act_min = act_min
        self.scale_mode = scale_mode
        self.accu_en = accu_en
        self.req_en = req_en
        self.bias_en = bias_en
        self.ping = ping
        self.k_size = k_size
        self.cim_sic= cim_sic 
        self.cim_soc = cim_soc
        self.sp_group = sp_group
        self.overwrite = overwrite
        self.align_mode=align_mode
        self.cim_ic= cim_ic # debug usage
        self.cim_oc = cim_oc # debug usage
        self.i_tile_coord = i_tile_coord # debug usage
        self.oc_group=oc_group # debug usage
        self.o_tile_coord=o_tile_coord # debug usage
        
class SanityCheckOp():
    def __init__(self, tile_attrs):
        self.tile_attrs = tile_attrs