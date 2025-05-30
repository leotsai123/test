from .op import MacroOp
from .isa_def import MacroOpType


class ExtSyncMacroOp(MacroOp):
    def __init__(self):
        super().__init__(MacroOpType.EXT_SYNC, False)

    def emit(self, emitter):
        emitter.emit_ext_sync()
