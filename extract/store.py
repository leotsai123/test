from .isa_def import Region, XY, XYZ, MacroOpType, ComponentType, DependencyTag
from .dma import DmaMacroOp


class StoreMacroOp(DmaMacroOp):
    def __init__(
        self,
        len_xyz=XYZ(0, 0, 0),
        src_region_type=Region.I_SRAM,
        src_region=[0, 0],
        src_base=0,
        dst_region_type=Region.DRAM,
        dst_region=[0, 0],
        dst_base=0,
        src_stride=XY(0, 0),
        dst_stride=XY(0, 0),
        transpose=False,
        dep_tag=None,
    ):
        super().__init__(
            macro_op_type=MacroOpType.STORE,
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
        self.transpose = transpose
        self.dep_tag = dep_tag

    def emit(self, emitter):
        emitter.set_target(ComponentType.STORE)
        dma_config = super().get_config()
        store_config_reg = emitter.store_config_reg
        store_config_reg.dma_config_reg.emit(emitter, dma_config)
        emitter.emit_store(self.transpose, self.dep_tag)
