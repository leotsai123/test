from .isa_def import MacroOpType, Region, XY, XYZ, ComponentType, DependencyTag
from .dma import DmaMacroOp


class LoadMacroOp(DmaMacroOp):
    def __init__(
        self,
        len_xyz=XYZ(0, 0, 0),
        src_region_type=Region.DRAM,
        src_region=[0, 0],
        src_base=0,
        dst_region_type=Region.I_SRAM,
        dst_region=[0, 0],
        dst_base=0,
        src_stride=XY(0, 0),
        dst_stride=XY(0, 0),
        dep_tag=None,
    ):
        super().__init__(
            macro_op_type=MacroOpType.LOAD,
            len_xyz=len_xyz,
            src_region_type=src_region_type,
            src_region=src_region,
            src_base=src_base,
            dst_region_type=dst_region_type,
            dst_region=dst_region,
            dst_base=dst_base,
            src_stride=src_stride,
            dst_stride=dst_stride,
        )
        self.dep_tag = dep_tag

    def emit(self, emitter):
        emitter.set_target(ComponentType.LOAD)
        dma_config = super().get_config()
        load_config_reg = emitter.load_config_reg
        load_config_reg.dma_config_reg.emit(emitter, dma_config)
        emitter.emit_load(self.dep_tag)
