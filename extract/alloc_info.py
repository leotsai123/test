from typing import Any
from tvm.relay.expr_functor import ExprVisitor
from tvm.relay import (
    Function,
)
from tvm.relay.backend.contrib.novella.utils import is_novella_func
from math import prod

class AllocInfo(ExprVisitor):
    def __init__(self, node_addr_map):
        super().__init__()
        self.input_offset = -1
        self.output_offset = -1
        self.is_valid_node = True
        self.node_addr_map = node_addr_map
        self.max_alloc_size = 0

    def visit_call(self, call):
        if isinstance(call.op, Function):
            self.visit(call.op)
            if self.is_valid_node:
                if self.output_offset == -1:
                    self.output_offset = self.node_addr_map[call]
                self.max_alloc_size = max(self.max_alloc_size, self.node_addr_map[call] + prod(call.checked_type.shape))
                # print(f"curent allocated size: {self.node_addr_map[call] + prod(call.checked_type.shape)}")
                
            self.is_valid_node = True
            for arg in call.args:
                self.visit(arg)
        else:
            if call.op.name == "reshape":
                self.is_valid_node = False

    
    def __call__(self, mod):
        for _, func in mod.functions.items():
            if is_novella_func(func):
                ifm = func.params[0]
                self.input_offset = self.node_addr_map[ifm]
                self.visit(func)
        return self.input_offset, self.output_offset, self.max_alloc_size
                

        
