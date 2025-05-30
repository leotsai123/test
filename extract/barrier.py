from .op import MacroOp
from .isa_def import MacroOpType


class BarrierMacroOp(MacroOp):
    def __init__(self):
        super().__init__(MacroOpType.BARRIER, False)

    def emit(self, emitter):
        emitter.emit_barrier()
