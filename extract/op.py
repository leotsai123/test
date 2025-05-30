from abc import ABC, abstractmethod


class MacroOp(ABC):
    def __init__(self, op_type, is_window_op):
        self.op_type = op_type
        self.is_window_op = is_window_op

    def __str__(self):
        var_list = {
            k: v for k, v in vars(self).items() if not k.startswith("_") and not callable(v)
        }

        return "MacroOp({})".format(var_list)

    @abstractmethod
    def emit(self, emitter):
        pass
