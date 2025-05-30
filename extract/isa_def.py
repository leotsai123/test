from enum import Enum, auto


class PoolOpType(Enum):
    AVG_POOL = 0
    MAX_POOL = 1


class BinEltwiseOpType(Enum):
    ADD = 0
    SUB = 1
    MUL = 2
    SHR = 3
    SHL = 4


class UnEltwiseOpType(Enum):
    CLZ = 0
    REQ = 1
    ADD_IMM = 2
    SUB_IMM_VAR = 3
    SUB_VAR_IMM = 4
    SHR_VAR_IMM = 5
    SHR_IMM_VAR = 6
    SHL_VAR_IMM = 7
    SHL_IMM_VAR = 8


class ReduceOpType(Enum):
    SUM = 0
    MAX = 1


class ComponentType(Enum):
    CU = 0
    PPU = 1
    LOAD = 2
    STORE = 3


class MacroOpType(Enum):
    EXT_SYNC = 0x0
    CONV = 0x1
    DW_CONV = 0x2
    POOL = 0x3
    BIN_ELTWISE = 0x4
    UN_ELTWISE = 0x5
    REDUCE = 0x6
    LOAD = 0x7
    STORE = 0x8
    BARRIER = 0xE
    SET_TARGET = 0xF


class Region(Enum):
    DRAM = 0
    I_SRAM = 1
    O_SRAM = 2
    LUT_MEM = 3


class ScaleModeType(Enum):
    PER_TENSOR_POT = 0x00
    PER_TENSOR_AFFINE = 0x10
    PER_CHANNEL_POT = 0x01
    PER_CHANNEL_AFFINE = 0x11


class ActivationType(Enum):
    NONE = 0
    CLIP = 1
    LUT = 2


class Dtype(Enum):
    INT8 = 0
    UINT8 = 1
    INT16 = 2
    UINT16 = 3
    INT24 = 4
    UINT24 = 5
    INT32 = 6
    UINT32 = 7


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


class HW(CommonConfig):
    def __init__(self, h, w):
        self.h = h
        self.w = w


class HWC(HW):
    def __init__(self, h, w, c):
        super().__init__(h, w)
        self.c = c


class XY(CommonConfig):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class XYZ(XY):
    def __init__(self, x, y, z):
        super().__init__(x, y)
        self.z = z


class TLBR(CommonConfig):
    def __init__(self, t, l, b, r):
        self.t = t
        self.l = l
        self.b = b
        self.r = r


class DependencyTag(CommonConfig):
    def __init__(self, ld_tag, ctu_tag, ppu_tag, st_tag):
        self.ld_tag = ld_tag
        self.ctu_tag = ctu_tag
        self.ppu_tag = ppu_tag
        self.st_tag = st_tag


class OpCode(Enum):
    EXT_SYNC = 0x00
    CONV = 0x01
    DW_CONV = 0x02
    POOL = 0x03
    BIN_ELTWISE = 0x04
    UN_ELTWISE = 0x05
    REDUCE = 0x06
    LOAD = 0x07
    STORE = 0x08
    BARRIER = 0x0E
    SET_TARGET = 0x0F

    IFM_HEIGHT = 0x10
    IFM_WIDTH = 0x11
    IFM_CHANNEL = 0x12
    IFM_PAD = 0x13
    IFM_UPSAMPLING_RATIO = 0x14
    IFM_ZERO_POINT = 0x15
    IFM_BASE = 0x90
    IFM2_BASE = 0x91
    IFM_REGION = 0x9E
    IFM2_REGION = 0x9F

    KERNEL_SHAPE = 0x20
    KERNEL_STRIDE = 0x21
    KERNEL_DILATION = 0x22
    WEIGHT_ZERO_POINT = 0x23
    TOTAL_IFM_CHANNEL = 0x24
    WEIGHT_BASE = 0xA0
    WEIGHT_REGION = 0xAF

    OFM_HEIGHT = 0x30
    OFM_WIDTH = 0x31
    OFM_CHANNEL = 0x32
    OFM_ZERO_POINT = 0x33
    OFM_BASE = 0xB0
    OFM_REGION = 0xBF

    QUANTIZATION_MODE = 0x40
    ACTIVATION_FUNCTION_TYPE = 0x41
    CLIP_MIN = 0x42
    CLIP_MAX = 0x43
    SCALE_MANTISSA = 0x44
    SCALE_SHIFT = 0x45
    BIAS_BASE = 0xC0
    SCALE_BASE = 0xC1

    X_LENGTH = 0x50
    Y_LENGTH = 0x51
    Z_LENGTH = 0x52
    SRC_BASE = 0xD0
    DST_BASE = 0xD1
    SRC_X_STRIDE = 0xD2
    DST_X_STRIDE = 0xD3
    SRC_Y_STRIDE = 0xD4
    DST_Y_STRIDE = 0xD5
    SRC_REGION = 0xD6
    DST_REGION = 0xD7

    DATA_TYPE = 0x60
    IMM_LOW = 0x61
    IMM_HIGH = 0x62

    def __str__(self):
        return self.name
