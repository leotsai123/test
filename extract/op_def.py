from enum import Enum, auto


class PoolOpType(Enum):
    MAX_POOL = auto()
    AVG_POOL = auto()


class EltwiseOpType(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    CLZ = auto()
    SHL = auto()
    SRL = auto()


class ReduceOpType(Enum):
    SUM = auto()
    MAX = auto()


class ComponentType(Enum):
    CU = auto()
    PPU = auto()
    DMA = auto()


class MacroOpType(Enum):
    EXT_SYNC = auto()
    CONV = auto()
    DW_CONV = auto()
    POOL = auto()
    BIN_ELTWISE = auto()
    UN_ELTWISE = auto()
    REDUCE = auto()
    DMA = auto()
    BARRIER = auto()
    SET_TARGET = auto()


class Region(Enum):
    DRAM = auto()
    W_SRAM = auto()
    I_SRAM = auto()
    O_SRAM = auto()
    LUT_MEM = auto()


class ScaleModeType(Enum):
    PER_TENSOR_POT = auto()
    PER_TENSOR_AFFINE = auto()
    PER_CHANNEL_POT = auto()
    PER_CHANNEL_AFFINE = auto()


class ActivationType(Enum):
    NONE_CLIP = auto()
    LUT = auto()


class Dtype(Enum):
    UINT8 = auto()
    INT8 = auto()
    INT16 = auto()
    INT24 = auto()
    INT32 = auto()
    DONT_CARE = 0xF


class CommonConfig:
    def __str__(self):
        var_list = {
            k: v for k, v in vars(self).items() if not k.startswith("_") and not callable(v)
        }
        return "{}({})".format(self.__class__.__name__, var_list)

    def __repr__(self):
        return self.__str__()


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
