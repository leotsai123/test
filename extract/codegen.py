import io
import math
from tvm.relay.backend.contrib.novella.utils import is_novella_func
from tvm.runtime.contrib.novella.utils import get_artifacts_dir, get_result_dir

# TODO: Remove result_dir


class CodeGen:
    def __init__(self):
        self.cpp_code_stream = io.StringIO()
        self.indent_level = 0

    def gen(self, code):
        self.cpp_code_stream.write(code)

    def enter_scope(self):
        self.indent_level += 1

    def exit_scope(self):
        self.indent_level -= 1

    def indent(self):
        self.cpp_code_stream.write("  " * self.indent_level)

    def gen_mem_data(self, const_stream, mem_size):
        # Remove dumping to result_dir
        with open(f"{get_artifacts_dir()}/const_data.txt", "w", encoding="utf-8") as f:
            f.write(const_stream)
        self.gen(f"uint8_t mem_data[{mem_size}] = {{}};\n")
        self.gen("uint8_t const_data[] = {")
        self.gen(f"{const_stream}")
        self.gen("};\n\n")

    def gen_varible(self, dtype, name, val=None):
        self.indent()
        if val is not None:
            self.gen(f"{dtype} {name} = {val};\n")
        else:
            self.gen(f"{dtype} {name};\n")

    def gen_insts(self, insts):
        self.gen("const uint8_t insts[] = {")
        inst_stream = io.StringIO()
        for inst in insts:
            inst_stream.write(f"0x{inst & 0xFF:02x}, ")
            inst_stream.write(f"0x{inst >> 8 & 0xFF:02x}, ")
            inst_stream.write(f"0x{inst >> 16 & 0xFF:02x}, ")
            inst_stream.write(f"0x{inst >> 24 & 0xFF:02x}, ")
        self.gen(inst_stream.getvalue())
        self.gen("};\n\n")

        with open(f"{get_artifacts_dir()}/insts.txt", "w", encoding="utf-8") as f:
            f.write(inst_stream.getvalue())

    def get_cpp_code(self):
        cpp_code = self.cpp_code_stream.getvalue()
        self.cpp_code_stream.close()
        return cpp_code

    def __call__(self, mod, insts, const_stream, const_size, mem_size, io_offset):
        self.gen('#include "tvm/runtime/c_runtime_api.h"\n')
        self.gen("#include <cstdlib>\n")
        self.gen("#include <cstdio>\n")
        self.gen("#include <cstring>\n")
        self.gen('#include "../src/runtime/contrib/novella/inst_runtime.hpp"\n')
        self.gen("namespace tvm {\n")
        self.gen("namespace runtime {\n")
        self.gen("#ifdef __cplusplus\n")
        self.gen('extern "C" {\n')
        self.gen("#endif\n\n")

        self.gen_mem_data(const_stream, mem_size)
        self.gen_insts(insts)

        for gv, func in mod.functions.items():
            if is_novella_func(func):
                func_name = gv.name_hint
                ifm = func.params[0]
                input_dtype = ifm.checked_type.dtype
                assert input_dtype in [
                    "int8",
                    "uint8",
                ], f"Novella only supports int8 or uint8 input: {input_dtype}"
                ofm = func.body
                output_dtype = ofm.checked_type.dtype
                break

        self.gen(f"TVM_DLL int {func_name}(\n")
        self.enter_scope()
        self.indent()
        self.gen("TVMValue* args,\n")
        self.indent()
        self.gen("int* type_codes,\n")
        self.indent()
        self.gen("int num_args,\n")
        self.indent()
        self.gen("TVMValue* out_value,\n")
        self.indent()
        self.gen("int* out_type_code\n")
        self.exit_scope()
        self.gen(") {\n")
        self.enter_scope()

        # Gen variables
        insts_len_name = "insts_len"
        self.gen_varible("uint32_t", insts_len_name, len(insts))

        input_offset_name = "input_offset"
        self.gen_varible("uint32_t", input_offset_name, io_offset[0])

        output_offset_name = "output_offset"
        self.gen_varible("uint32_t", output_offset_name, io_offset[1])

        mem_size_name = "mem_size"
        self.gen_varible("uint32_t", mem_size_name, mem_size)

        const_size_name = "const_size"
        self.gen_varible("uint32_t", const_size_name, const_size)

        ifm_size_name = "ifm_size"
        self.gen_varible("uint32_t", ifm_size_name, math.prod(ifm.checked_type.shape))

        ofm_size_name = "ofm_size"
        self.gen_varible("uint32_t", ofm_size_name, math.prod(ofm.checked_type.shape))

        # Gen const data
        self.indent()
        self.gen(f"memcpy(mem_data, const_data, {const_size_name});\n")

        # Gen Ifm
        self.indent()
        self.gen(
            f"auto {ifm.name_hint} = reinterpret_cast<{input_dtype}_t*>((reinterpret_cast<DLTensor*>(args[0].v_handle)->data));\n"
        )

        self.indent()
        self.gen(f"memcpy(mem_data + {input_offset_name}, {ifm.name_hint}, {ifm_size_name});\n")


        iss_attr_name = "iss_attr"
        self.indent()
        self.gen(f"novella::IssRuntimeAttr {iss_attr_name} = {{ \"{get_result_dir()}\" }};\n")

        self.indent()
        self.gen(
            f"novella::RunInst(&insts[0], {insts_len_name}, &mem_data[0], {mem_size_name}, {input_offset_name}, {output_offset_name}, {ofm_size_name}, {iss_attr_name});\n"
        )

        # Gen Ofm
        self.indent()
        self.gen(
            f"auto output = reinterpret_cast<{output_dtype}_t*>((reinterpret_cast<DLTensor*>(args[1].v_handle)->data));\n"
        )

        self.indent()
        self.gen(f"memcpy(output, mem_data + {output_offset_name}, {ofm_size_name});\n\n")

        self.indent()
        self.gen("return 0;\n")
        self.exit_scope()
        self.gen("}\n")
        self.gen("#ifdef __cplusplus\n")
        self.gen("}\n")
        self.gen("#endif\n")
        self.gen("} // namespace tvm\n")
        self.gen("} // namespace runtime\n")
        return self.get_cpp_code()
