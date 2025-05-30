from .op import MacroOp
from .isa_def import Region, XY, XYZ, OpCode


class DmaMacroOp(MacroOp):
    def __init__(
        self,
        macro_op_type,
        len_xyz=XYZ(0, 0, 0),
        src_region_type=Region.DRAM,
        src_region=[0, 0],
        src_base=0,
        dst_region_type=Region.DRAM,
        dst_region=[0, 0],
        dst_base=0,
        src_stride=XY(0, 0),
        dst_stride=XY(0, 0),
    ):
        super().__init__(macro_op_type, False)
        self.len_xyz = len_xyz
        self.src_region_type = src_region_type
        self.src_region = src_region
        self.src_base = src_base
        self.dst_region_type = dst_region_type
        self.dst_region = dst_region
        self.dst_base = dst_base
        self.src_stride = src_stride
        self.dst_stride = dst_stride

    def get_config(self):
        dma_config = {
            OpCode.X_LENGTH: self.len_xyz.x,
            OpCode.Y_LENGTH: self.len_xyz.y,
            OpCode.Z_LENGTH: self.len_xyz.z,
            OpCode.SRC_BASE: self.src_base,
            OpCode.DST_BASE: self.dst_base,
            OpCode.SRC_X_STRIDE: self.src_stride.x,
            OpCode.DST_X_STRIDE: self.dst_stride.x,
            OpCode.SRC_Y_STRIDE: self.src_stride.y,
            OpCode.DST_Y_STRIDE: self.dst_stride.y,
            OpCode.SRC_REGION: [self.src_region_type, self.src_region],
            OpCode.DST_REGION: [self.dst_region_type, self.dst_region],
        }

        return dma_config
