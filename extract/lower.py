from os import waitpid
from typing import List, Dict, Tuple, Optional, Union, Set, Any
from tvm import IRModule
from tvm.relay import (
    Constant,
    Call,
    Function,
    Var,
)
from tvm.ir.container import Map
from tvm.relay.expr_functor import ExprVisitor
from tvm.relay.backend.contrib.novella.op.op_attrs import (
    Q_CONV2D_OP_NAME,
    Q_DW_CONV2D_OP_NAME,
    Q_POOL2D_OP_NAME,
    BIN_ELTWISE_OP_NAME,
    BIN_IMM_ELTWISE_OP_NAME,
    UNARY_ELTWISE_OP_NAME,
    REDUCE_OP_NAME,
    Q_SOFTMAX_OP_NAME,
    Q_LAYER_NORM_OP_NAME,
    Q_GELU_OP_NAME,
    IDENTITY_OP_NAME,
)
from tvm.relay.backend.contrib.novella.utils import (
    is_novella_func,
    Conv2dArgs,
    DwConv2dArgs,
    IdentityArgs,
    BinEltwiseArgs,
    BinImmEltwiseArgs,
    UnaryEltwiseArgs,
    Pool2dArgs,
    LayerNormArgs,
    SoftmaxArgs,
    GELUArgs,
    ReduceArgs,
    ReqArgs,
)
from tvm.contrib.novella import (
    MacroOp,
    ConvMacroOp,
    DwConvMacroOp,
    LoadMacroOp,
    StoreMacroOp,
    BinEltwiseMacroOp,
    UnEltwiseMacroOp,
    PoolMacroOp,
    ReduceMacroOp,
    WindowInfo,
    BarrierMacroOp,
    ExtSyncMacroOp,
)
from tvm.contrib.novella.isa_def import (
    HWC,
    HW,
    Region,
    Dtype,
    TLBR,
    ActivationType,
    ScaleModeType,
    XYZ,
    XY,
    PoolOpType,
    BinEltwiseOpType,
    UnEltwiseOpType,
    ReduceOpType,
    DependencyTag,
)
from math import frexp
from tvm.runtime.contrib.novella.utils import get_artifacts_dir

CH_GROUP_SIZE = 8
CTU_REDUCED_SIZE = 32
ISRAM_BANK_GROUP_SIZE = 1
PSUM_CH_SIZE = 16
PSUM_BANK_GROUP_SIZE = 4


def get_fp_data(scale):
    m, e = frexp(scale)
    m_bits = 16
    e_bits = 8
    while m < 2 ** m_bits and (-e) < 2**e_bits:
        m = m * 2
        e = e - 1
    m = m / 2
    e = e + 1
    shamt = (-e) & 0xFF
    mantissa = int(round(m)) & 0xFFFF

    return shamt, mantissa


class MacroOpLower(ExprVisitor):
    def __init__(
        self,
        node_tile_attrs_map,
        node_ifm_callnode_map,
        node_addr_map,
        const_addr_map,
        const_stream,
    ):
        ExprVisitor.__init__(self)
        self.node_tile_attrs_map = node_tile_attrs_map
        self.node_ifm_callnode_map = node_ifm_callnode_map
        self.node_addr_map = node_addr_map
        self.const_addr_map = const_addr_map
        self.const_stream = const_stream
        self.cur_visited_calls = []
        self.macro_ops = []
        self.num_macro_ops = 0
        self.num_conv_macro_ops = 0
        self.num_dw_conv_macro_ops = 0
        self.num_pool_macro_ops = 0
        self.num_bin_eltwise_macro_ops = 0
        self.num_reduce_macro_ops = 0
        self.num_un_eltwise_macro_ops = 0
        self.num_load_macro_ops = 0
        self.num_store_macro_ops = 0
        self.cur_src_addr_base = -1
        self.cur_isram_bank_start = 0
        self.cur_psum_buffer_idx = 0
        self.cur_psum_allocated_size = 0
        self.cur_psum_allocation_map = {}
        self.ld_dep_tag = 0
        self.st_dep_tag = 0
        self.ctu_dep_tag = 0
        self.ppu_dep_tag = 0
        self.ld_op_list = []
        self.st_op_list = []
        self.ctu_op_list = []
        self.ppu_op_list = []
        self.barrier_nodes = []

    def visit_call(self, call):
        for a in call.args:
            self.visit(a)
        if isinstance(call.op, Function):
            self.visit(call.op)
            if len(self.cur_visited_calls) == 0:
                return
            tile_nums = len(self.node_tile_attrs_map[self.cur_visited_calls[0]])
            # TODO: memory operation is ignored here
            barrier_node_name = (
                self.cur_visited_calls[-1].attrs.node_name
                if hasattr(self.cur_visited_calls[-1].attrs, "node_name")
                else "x"
            )
            self.barrier_nodes.append(barrier_node_name)

            for tile_idx in range(tile_nums):
                for visited_call in self.cur_visited_calls:
                    tile_attr = self.node_tile_attrs_map[visited_call][tile_idx]
                    src_addr_bases = []
                    if visited_call.args[0] in list(call.op.params):
                        src_addr_bases.append(
                            self.node_addr_map[
                                call.args[list(call.op.params).index(visited_call.args[0])]
                            ]
                            + tile_attr["tile_offset"]
                        )
                    node_addr_base = (
                        tile_attr["o_tile_offset"]
                        if tile_attr["o_tile_offset"] == -1
                        else self.node_addr_map[call] + tile_attr["o_tile_offset"]
                    )
                    self.gen_macro_ops(visited_call, node_addr_base, src_addr_bases, tile_attr)
                self.cur_psum_buffer_idx = (
                    self.cur_psum_buffer_idx + 1
                ) % 2  # update psum buffer index
                self.cur_psum_allocated_size = 0
                self.cur_psum_allocation_map = {}
            self.cur_visited_calls = []
            self.ld_op_list = []
            self.st_op_list = []
            self.ctu_op_list = []
            self.ppu_op_list = []
            self.ld_dep_tag = 0
            self.st_dep_tag = 0
            self.ctu_dep_tag = 0
            self.ppu_dep_tag = 0
            self.macro_ops.append(BarrierMacroOp())  # add barrier after each fusion
        else:
            if call.op.name == "reshape":
                return
            self.cur_visited_calls.append(call)

    def gen_macro_ops(self, call_node, node_addr_base, src_addr_bases, tile_attr):
        # print(
        #     f"op name: {op.call_node.op.name}, op.output_tile_shape: {op.output_tile_shape}"
        # )
        self.cur_isram_bank_start = 0
        isram_cache_hit = (
            call_node.op.name == Q_CONV2D_OP_NAME and self.cur_src_addr_base == src_addr_bases[0]
        )
        if tile_attr["from_dram"] and not isram_cache_hit:
            if call_node.op.name == Q_CONV2D_OP_NAME:
                reduce_group_size = CTU_REDUCED_SIZE
                dst_region_type = Region.I_SRAM
            else:
                reduce_group_size = PSUM_CH_SIZE
                dst_region_type = Region.O_SRAM
            reduced_group_num = 0
            reduce_groups_num = int(tile_attr["tile_shape"][2] // reduce_group_size)
            has_remaind_ch = (
                True if tile_attr["tile_shape"][2] % int(reduce_group_size) != 0 else False
            )

            # Handle the reduce group size
            while reduced_group_num < reduce_groups_num:
                for ch_group in range((reduce_group_size // CH_GROUP_SIZE)):
                    dst_region = (
                        [
                            self.cur_isram_bank_start * ISRAM_BANK_GROUP_SIZE,
                            self.cur_isram_bank_start * ISRAM_BANK_GROUP_SIZE + 1,
                        ]
                        if call_node.op.name == Q_CONV2D_OP_NAME
                        else [
                            self.cur_psum_buffer_idx * PSUM_BANK_GROUP_SIZE,
                            self.cur_psum_buffer_idx * PSUM_BANK_GROUP_SIZE + 1,
                        ]
                    )
                    self.macro_ops.append(
                        self.gen_load_macro_op(
                            reduced_group_num % 2,
                            call_node,
                            tile_attr,
                            Region.DRAM,
                            [0, 0],  # Don't care
                            src_addr_bases[0]
                            + (reduced_group_num * reduce_group_size + ch_group * CH_GROUP_SIZE)
                            * call_node.args[0].checked_type.shape[1]
                            * call_node.args[0].checked_type.shape[2],
                            dst_region_type,
                            dst_region,
                            reduce_group_size
                            * int(reduced_group_num // 2)
                            * tile_attr["tile_shape"][0]
                            * tile_attr["tile_shape"][1]
                            + ch_group * CH_GROUP_SIZE,
                            CH_GROUP_SIZE,
                        )
                    )
                    self.ld_op_list.append(self.macro_ops[-1])
                is_last = (
                    reduced_group_num == reduce_groups_num - 1 and not has_remaind_ch
                )  # check if required to requantize
                self.macro_ops.append(
                    self.gen_macro_op(reduced_group_num, call_node, True, is_last, 0, tile_attr)
                )
                if call_node.op.name == Q_CONV2D_OP_NAME:  # update ISRAM bank start
                    self.cur_isram_bank_start = (self.cur_isram_bank_start + 1) % 2
                reduced_group_num += 1
            ch_group_idx = 0
            dst_region = (
                [self.cur_isram_bank_start, self.cur_isram_bank_start + 1]
                if call_node.op.name == Q_CONV2D_OP_NAME
                else [
                    self.cur_psum_buffer_idx * PSUM_BANK_GROUP_SIZE,
                    self.cur_psum_buffer_idx * PSUM_BANK_GROUP_SIZE + 1,
                ]
            )

            # Handle the channel group size
            while ch_group_idx < int(
                (tile_attr["tile_shape"][2] % reduce_group_size) // CH_GROUP_SIZE
            ):
                self.macro_ops.append(
                    self.gen_load_macro_op(
                        reduced_group_num % 2,
                        call_node,
                        tile_attr,
                        Region.DRAM,
                        [0, 0],  # Don't care
                        src_addr_bases[0]
                        + (reduced_group_num * reduce_group_size + ch_group_idx * CH_GROUP_SIZE)
                        * call_node.args[0].checked_type.shape[1]
                        * call_node.args[0].checked_type.shape[2],
                        dst_region_type,
                        [dst_region[0], dst_region[1]],
                        reduce_group_size
                        * int(reduced_group_num // 2)
                        * tile_attr["tile_shape"][0]
                        * tile_attr["tile_shape"][1]
                        + ch_group_idx * CH_GROUP_SIZE,
                        CH_GROUP_SIZE,
                    )
                )
                self.ld_op_list.append(self.macro_ops[-1])
                ch_group_idx += 1

            # Handle the remaining channel
            if (tile_attr["tile_shape"][2] % reduce_group_size) % CH_GROUP_SIZE != 0:
                self.macro_ops.append(
                    self.gen_load_macro_op(
                        reduced_group_num % 2,
                        call_node,
                        tile_attr,
                        Region.DRAM,
                        [0, 0],  # Don't care
                        src_addr_bases[0]
                        + (reduced_group_num * reduce_group_size + ch_group_idx * CH_GROUP_SIZE)
                        * call_node.args[0].checked_type.shape[1]
                        * call_node.args[0].checked_type.shape[2],
                        dst_region_type,
                        [dst_region[0], dst_region[1]],
                        reduce_group_size
                        * int(reduced_group_num // 2)
                        * tile_attr["tile_shape"][0]
                        * tile_attr["tile_shape"][1]
                        + ch_group_idx * CH_GROUP_SIZE,
                        (tile_attr["tile_shape"][2] % reduce_group_size) % CH_GROUP_SIZE,
                    )
                )
                self.ld_op_list.append(self.macro_ops[-1])
            remand_ch = ch_group_idx * CH_GROUP_SIZE + (
                (tile_attr["tile_shape"][2] % reduce_group_size) % CH_GROUP_SIZE
            )
            if remand_ch != 0:
                self.macro_ops.append(
                    self.gen_macro_op(
                        reduced_group_num, call_node, True, True, remand_ch, tile_attr
                    )
                )
        else:
            self.macro_ops.append(self.gen_macro_op(0, call_node, False, True, 0, tile_attr))

        # Update current isram source address base and allocate psum buffer
        if tile_attr["from_dram"] and call_node.op.name == Q_CONV2D_OP_NAME:
            self.cur_src_addr_base = src_addr_bases[0]
            if call_node not in self.cur_psum_allocation_map:
                self.cur_psum_allocation_map[call_node] = [
                    self.cur_psum_allocated_size,
                    self.cur_psum_allocated_size
                    + tile_attr["o_tile_shape"][0] * tile_attr["o_tile_shape"][1],
                ]
                self.cur_psum_allocated_size += (
                    tile_attr["o_tile_shape"][0] * tile_attr["o_tile_shape"][1]
                )
        else:
            self.cur_psum_allocation_map[call_node] = self.cur_psum_allocation_map[
                call_node.args[0]
            ]
            del self.cur_psum_allocation_map[call_node.args[0]]

        if tile_attr["to_dram"]:
            idx = 0
            while idx < int(tile_attr["o_tile_shape"][2] // CH_GROUP_SIZE):
                src_region = [
                    self.cur_psum_buffer_idx * PSUM_BANK_GROUP_SIZE,
                    self.cur_psum_buffer_idx * PSUM_BANK_GROUP_SIZE + 1,
                ]
                self.macro_ops.append(
                    self.gen_store_macro_op(
                        idx,
                        call_node,
                        tile_attr,
                        Region.O_SRAM,
                        src_region,
                        self.cur_psum_allocation_map[call_node][0] + idx * CH_GROUP_SIZE,
                        Region.DRAM,
                        [0, 0],  # Don't care
                        node_addr_base
                        + idx
                        * call_node.checked_type.shape[1]
                        * call_node.checked_type.shape[2]
                        * CH_GROUP_SIZE,
                        CH_GROUP_SIZE,
                    )
                )
                self.st_op_list.append(self.macro_ops[-1])
                idx += 1
            if tile_attr["o_tile_shape"][2] % CH_GROUP_SIZE != 0:
                src_region = [
                    self.cur_psum_buffer_idx * PSUM_BANK_GROUP_SIZE,
                    self.cur_psum_buffer_idx * PSUM_BANK_GROUP_SIZE + 1,
                ]
                self.macro_ops.append(
                    self.gen_store_macro_op(
                        idx,
                        call_node,
                        tile_attr,
                        Region.O_SRAM,
                        src_region,
                        self.cur_psum_allocation_map[call_node][0] + idx * CH_GROUP_SIZE,
                        Region.DRAM,
                        [0, 0],  # Don't care
                        node_addr_base
                        + idx
                        * call_node.checked_type.shape[1]
                        * call_node.checked_type.shape[2]
                        * CH_GROUP_SIZE,
                        tile_attr["o_tile_shape"][2] % CH_GROUP_SIZE,
                    )
                )
                self.st_op_list.append(self.macro_ops[-1])

    def gen_macro_op(self, idx, call_node, is_slice, is_last, remand_ch, tile_attr):
        # Allocate psum buffer
        if tile_attr["from_dram"] and call_node.op.name != Q_CONV2D_OP_NAME:
            if call_node.args[0] not in self.cur_psum_allocation_map:
                self.cur_psum_allocation_map[call_node.args[0]] = [
                    self.cur_psum_allocated_size,
                    self.cur_psum_allocated_size
                    + tile_attr["o_tile_shape"][0] * tile_attr["o_tile_shape"][1],
                ]
                self.cur_psum_allocated_size += (
                    tile_attr["o_tile_shape"][0] * tile_attr["o_tile_shape"][1]
                )

        op_name = call_node.op.name
        if op_name == Q_CONV2D_OP_NAME:
            conv_macro_op = self.gen_conv_macro_op(
                idx, call_node, is_slice, is_last, remand_ch, tile_attr
            )
            self.ctu_op_list.append(conv_macro_op)
            return conv_macro_op
        if op_name == Q_DW_CONV2D_OP_NAME:
            dw_conv_macro_op = self.gen_dwconv_macro_op(idx, call_node, is_slice, tile_attr)
            self.ppu_op_list.append(dw_conv_macro_op)
            return dw_conv_macro_op
        if op_name == Q_POOL2D_OP_NAME:
            pool_macro_op = self.gen_pool_macro_op(call_node, idx, is_last, tile_attr)
            self.ppu_op_list.append(pool_macro_op)
            return pool_macro_op
        if (
            op_name == "qnn.requantize"
            or op_name == UNARY_ELTWISE_OP_NAME
            or op_name == BIN_IMM_ELTWISE_OP_NAME
        ):
            un_eltwise_macro_op = self.gen_un_eltwise_macro_op(call_node, op_name, tile_attr)
            self.ppu_op_list.append(un_eltwise_macro_op)
            return un_eltwise_macro_op
        if op_name == BIN_ELTWISE_OP_NAME:
            bin_eltwise_macro_op = self.gen_bin_eltwise_macro_op(call_node, tile_attr)
            self.ppu_op_list.append(bin_eltwise_macro_op)
            return bin_eltwise_macro_op
        if op_name == REDUCE_OP_NAME:
            reduce_macro_op = self.gen_reduce_macro_op(call_node, tile_attr)
            self.ppu_op_list.append(reduce_macro_op)
            return reduce_macro_op
        # else:
        #     assert False, f"Unsupported op: {op_name}"

    def gen_conv_macro_op(self, idx, call_node, is_slice, is_last, remand_ch, tile_attr):
        ifm = call_node.args[Conv2dArgs.kIfmIdx.value]
        weight = call_node.args[Conv2dArgs.kWeightIdx.value]
        bias = call_node.args[Conv2dArgs.kBiasIdx.value]
        scale = call_node.args[Conv2dArgs.kScaleIdx.value]
        bias_base = self.const_addr_map[bias] + tile_attr["bias_offset"]
        if tile_attr["scale_offset"] == -1:
            scale_base = -1
            scale_shift, scale_mantissa = get_fp_data(scale.data.asnumpy())
        else:
            scale_base = self.const_addr_map[scale] + tile_attr["scale_offset"]
            scale_shift = 0
            scale_mantissa = 0
        ifm_shape = HWC(
            tile_attr["tile_shape"][0],
            tile_attr["tile_shape"][1],
            tile_attr["tile_shape"][2],
        )  # HWC
        if is_slice:
            ifm_sram_base = (idx // 2) * tile_attr["tile_shape"][0] * tile_attr["tile_shape"][1]
            ifm_bank_start = idx % 2
            ifm_bank_span = 1
            ifm_shape.c = remand_ch if remand_ch != 0 else CTU_REDUCED_SIZE
        else:
            ifm_sram_base = 0
            ifm_bank_start = 0
            ifm_bank_span = 2
        ifm_dtype = None
        if ifm.checked_type.dtype == "int8":
            ifm_dtype = Dtype.INT8
        elif ifm.checked_type.dtype == "uint8":
            ifm_dtype = Dtype.UINT8
        else:
            assert False, f"Unsupported ifm dtype: {ifm.checked_type.dtype}"
        ifm_zp = call_node.attrs.ifm_zp

        padding = TLBR(
            t=tile_attr["padding"][0],
            l=tile_attr["padding"][1],
            b=tile_attr["padding"][2],
            r=tile_attr["padding"][3],
        )
        kernel_shape = HW(
            call_node.attrs.kernel_shape[0],
            call_node.attrs.kernel_shape[1],
        )
        strides = HW(
            call_node.attrs.strides[0],
            call_node.attrs.strides[1],
        )
        dilation = HW(
            call_node.attrs.dilation[0],
            call_node.attrs.dilation[1],
        )
        window_info = WindowInfo(
            padding=padding,
            kernel_shape=kernel_shape,
            strides=strides,
            dilation=dilation,
        )
        weight_region_type = Region.DRAM
        weight_base = (
            self.const_addr_map[weight]
            + tile_attr["weight_offset"]
            + idx * CTU_REDUCED_SIZE * min(CH_GROUP_SIZE, tile_attr["o_tile_shape"][2])
        )
        weight_dtype = None
        if weight.checked_type.dtype == "int8":
            weight_dtype = Dtype.INT8
        elif weight.checked_type.dtype == "uint8":
            weight_dtype = Dtype.UINT8
        else:
            assert False, f"Unsupported weight dtype: {weight.checked_type.dtype}"
        weight_zp = call_node.attrs.weight_zp
        total_ic_num = ifm.checked_type.shape[3]

        ofm_shape = HWC(
            tile_attr["o_tile_shape"][0],
            tile_attr["o_tile_shape"][1],
            tile_attr["o_tile_shape"][2],
        )  # HWC
        ofm_zp = call_node.attrs.ofm_zp
        ofm_sram_base = self.cur_psum_allocated_size
        ofm_bank_start = self.cur_psum_buffer_idx * PSUM_BANK_GROUP_SIZE
        ofm_dtype = None
        if call_node.checked_type.dtype == "int8":
            ofm_dtype = Dtype.INT8
        elif call_node.checked_type.dtype == "uint8":
            ofm_dtype = Dtype.UINT8
        else:
            assert False, f"Unsupported ofm dtype: {call_node.checked_type.dtype}"

        act_type = call_node.attrs.activation
        if call_node.attrs.activation == "NONE":
            act_type = ActivationType.NONE
        elif call_node.attrs.activation == "CLIP":
            act_type = ActivationType.CLIP
        elif call_node.attrs.activation == "LUT":
            act_type = ActivationType.LUT
        else:
            assert False, f"Unsupported activation type: {call_node.attrs.activation}"

        req_en = True if is_last else False
        accu_en = True if idx != 0 else False
        bias_en = True if idx == 0 else False

        ofm_bank_span = 1 if req_en else 4

        ld_dep_tag = 0
        for ld_op in reversed(self.ld_op_list):
            if (
                (
                    ld_op.dst_region_type == Region.O_SRAM
                    and ld_op.dst_region[0] == ofm_bank_start
                    and ofm_sram_base <= ld_op.dst_base // CTU_REDUCED_SIZE
                    and ofm_sram_base + ofm_shape.h * ofm_shape.w
                    > ld_op.dst_base // CTU_REDUCED_SIZE
                )
                or (
                    ld_op.dst_region_type == Region.I_SRAM
                    and ld_op.dst_region[0] == ifm_bank_start
                    and ifm_sram_base <= ld_op.dst_base // CTU_REDUCED_SIZE
                    and ifm_sram_base + ifm_shape.h * ifm_shape.w
                    > ld_op.dst_base // CTU_REDUCED_SIZE
                )
                and ld_op.dep_tag.ld_tag - self.ld_dep_tag < 8
            ):
                ld_dep_tag = ld_op.dep_tag.ld_tag
                break

        st_dep_tag = 0
        for st_op in reversed(self.st_op_list):
            if st_op.src_region[0] == ofm_bank_start and st_op.dep_tag.st_tag - self.st_dep_tag < 8:
                st_dep_tag = st_op.dep_tag.st_tag
                break

        ppu_dep_tag = 0
        for ppu_op in reversed(self.ppu_op_list):
            if (
                ppu_op.ofm_bank_start == ofm_bank_start
                and ppu_op.ofm_sram_base < ofm_sram_base
                and ppu_op.ofm_sram_base + ppu_op.ofm_shape.h * ppu_op.ofm_shape.w > ofm_sram_base
                or ppu_op.ifm_bank_start == ofm_bank_start
                and ppu_op.ifm_sram_base < ofm_sram_base
                and ppu_op.ifm_sram_base + ppu_op.ofm_shape.h * ppu_op.ofm_shape.w > ofm_sram_base
                and self.ppu_dep_tag - ppu_op.dep_tag.ppu_tag < 8
            ):
                ppu_dep_tag = ppu_op.dep_tag.ppu_tag
                break

        ctu_dep_tag = 0
        if len(self.ctu_op_list) > 0 and isinstance(self.macro_ops[-1], ConvMacroOp):
            ctu_dep_tag = self.ctu_op_list[-1].dep_tag.ctu_tag
            ld_dep_tag = 0
            ppu_dep_tag = 0
            st_dep_tag = 0
        else:
            self.ctu_dep_tag += 1
            ctu_dep_tag = self.ctu_dep_tag

        self.num_conv_macro_ops += 1

        return ConvMacroOp(
            ofm_shape=ofm_shape,
            ifm_shape=ifm_shape,
            ifm_sram_base=ifm_sram_base,
            ifm_bank_start=ifm_bank_start,
            ifm_bank_span=ifm_bank_span,
            ifm_dtype=ifm_dtype,
            ifm_zp=ifm_zp,
            window_info=window_info,
            weight_region_type=weight_region_type,
            weight_base=weight_base,
            weight_dtype=weight_dtype,
            weight_zp=weight_zp,
            total_ic_num=total_ic_num,
            ofm_zp=ofm_zp,
            ofm_sram_base=ofm_sram_base,
            ofm_bank_start=ofm_bank_start,
            ofm_bank_span=ofm_bank_span,
            ofm_dtype=ofm_dtype,
            scale_mantissa=scale_mantissa,
            scale_shift=scale_shift,
            scale_base=scale_base,
            bias_base=bias_base,
            act_type=act_type,
            accu_en=accu_en,
            req_en=req_en,
            bias_en=bias_en,
            dep_tag=DependencyTag(ld_dep_tag, ctu_dep_tag, ppu_dep_tag, st_dep_tag),
        )

    def gen_dwconv_macro_op(self, idx, call_node, accu_en, tile_attr):  # TODO: add more parameters
        ifm = call_node.args[DwConv2dArgs.kIfmIdx.value]
        weight = call_node.args[DwConv2dArgs.kWeightIdx.value]
        bias = call_node.args[DwConv2dArgs.kBiasIdx.value]
        scale = call_node.args[DwConv2dArgs.kScaleIdx.value]
        bias_base = self.const_addr_map[bias] + tile_attr["bias_offset"]
        if tile_attr["scale_offset"] == -1:
            scale_base = -1
            scale_shift, scale_mantissa = get_fp_data(scale.data.asnumpy())
        else:
            scale_base = self.const_addr_map[scale] + tile_attr["scale_offset"]
            scale_shift = 0
            scale_mantissa = 0

        ifm_shape = HWC(
            tile_attr["tile_shape"][0],
            tile_attr["tile_shape"][1],
            tile_attr["tile_shape"][2],
        )  # HWC
        ifm_sram_base = self.cur_psum_allocation_map[ifm][0]
        ifm_bank_start = self.cur_psum_buffer_idx * PSUM_BANK_GROUP_SIZE
        ifm_dtype = None
        if ifm.checked_type.dtype == "int8":
            ifm_dtype = Dtype.INT8
        elif ifm.checked_type.dtype == "uint8":
            ifm_dtype = Dtype.UINT8
        else:
            assert False, f"Unsupported ifm dtype: {ifm.checked_type.dtype}"
        ifm_bank_span = 1 if ifm_dtype == Dtype.INT8 or ifm_dtype == Dtype.UINT8 else 4
        ifm_zp = call_node.attrs.ifm_zp

        padding = TLBR(
            t=tile_attr["padding"][0],
            l=tile_attr["padding"][1],
            b=tile_attr["padding"][2],
            r=tile_attr["padding"][3],
        )
        kernel_shape = HW(
            call_node.attrs.kernel_shape[0],
            call_node.attrs.kernel_shape[1],
        )
        strides = HW(
            call_node.attrs.strides[0],
            call_node.attrs.strides[1],
        )
        dilation = HW(
            call_node.attrs.dilation[0],
            call_node.attrs.dilation[1],
        )
        window_info = WindowInfo(
            padding=padding,
            kernel_shape=kernel_shape,
            strides=strides,
            dilation=dilation,
        )
        weight_region_type = Region.DRAM
        # weight_base = self.const_table[weight][0]
        weight_base = self.const_addr_map[weight] + tile_attr["weight_offset"]
        weight_dtype = None
        if weight.checked_type.dtype == "int8":
            weight_dtype = Dtype.INT8
        elif weight.checked_type.dtype == "uint8":
            weight_dtype = Dtype.UINT8
        else:
            assert False, f"Unsupported weight dtype: {weight.checked_type.dtype}"
        weight_zp = call_node.attrs.weight_zp
        total_ic_num = ifm.checked_type.shape[3]

        ofm_shape = HWC(
            tile_attr["o_tile_shape"][0],
            tile_attr["o_tile_shape"][1],
            tile_attr["o_tile_shape"][2],
        )
        ofm_zp = call_node.attrs.ofm_zp
        ofm_sram_base = ifm_sram_base
        ofm_bank_start = ifm_bank_start
        ofm_dtype = None
        if call_node.checked_type.dtype == "int8":
            ofm_dtype = Dtype.INT8
        elif call_node.checked_type.dtype == "uint8":
            ofm_dtype = Dtype.UINT8
        else:
            assert False, f"Unsupported ofm dtype: {call_node.checked_type.dtype}"
        ofm_bank_span = 1 if ofm_dtype == Dtype.INT8 or ofm_dtype == Dtype.UINT8 else 4

        act_type = call_node.attrs.activation
        if call_node.attrs.activation == "NONE":
            act_type = ActivationType.NONE
        elif call_node.attrs.activation == "CLIP":
            act_type = ActivationType.CLIP
        elif call_node.attrs.activation == "LUT":
            act_type = ActivationType.LUT
        else:
            assert False, f"Unsupported activation type: {call_node.attrs.activation}"

        accu_en = False
        req_en = True
        bias_en = True

        self.ppu_dep_tag += 1
        ppu_dep_tag = self.ppu_dep_tag
        ld_dep_tag = 0
        if self.ld_op_list[-1].dst_region_type == Region.O_SRAM:
            ld_dep_tag = self.ld_dep_tag

        ctu_dep_tag = 0
        for ctu_op in reversed(self.ctu_op_list):
            if (
                ctu_op.ofm_bank_start == ofm_bank_start
                and ctu_op.ofm_sram_base <= ofm_sram_base
                and ctu_op.ofm_sram_base + ctu_op.ofm_shape.h * ctu_op.ofm_shape.w > ofm_sram_base
                or ctu_op.ofm_bank_start == ofm_bank_start
                and ctu_op.ofm_sram_base <= ifm_sram_base
                and ctu_op.ofm_sram_base + ctu_op.ofm_shape.h * ctu_op.ofm_shape.w > ifm_sram_base
                and self.ppu_dep_tag - ctu_op.dep_tag.ppu_tag < 8
            ):
                ctu_dep_tag = ctu_op.dep_tag.ctu_tag
                break

        st_dep_tag = 0
        if len(self.st_op_list) > 0:
            for st_op in reversed(self.st_op_list):
                if (
                    st_op.src_region[0] == ofm_bank_start
                    and st_op.dep_tag.st_tag - self.st_dep_tag < 8
                ):
                    st_dep_tag = st_op.dep_tag.st_tag
                    break

        self.num_dw_conv_macro_ops += 1

        return DwConvMacroOp(
            ofm_shape=ofm_shape,
            ifm_shape=ifm_shape,
            ifm_sram_base=ifm_sram_base,
            ifm_bank_start=ifm_bank_start,
            ifm_bank_span=ifm_bank_span,
            ifm_dtype=ifm_dtype,
            ifm_zp=ifm_zp,
            window_info=window_info,
            weight_region_type=weight_region_type,
            weight_base=weight_base,
            weight_dtype=weight_dtype,
            weight_zp=weight_zp,
            total_ic_num=total_ic_num,
            ofm_zp=ofm_zp,
            ofm_sram_base=ofm_sram_base,
            ofm_bank_start=ofm_bank_start,
            ofm_bank_span=ofm_bank_span,
            ofm_dtype=ofm_dtype,
            scale_mantissa=scale_mantissa,
            scale_shift=scale_shift,
            scale_base=scale_base,
            bias_base=bias_base,
            act_type=act_type,
            accu_en=accu_en,
            req_en=req_en,
            bias_en=bias_en,
            dep_tag=DependencyTag(ld_dep_tag, ctu_dep_tag, ppu_dep_tag, st_dep_tag),
        )

    def gen_load_macro_op(
        self,
        idx,
        call_node,
        tile_attr,
        src_region_type,
        src_region,
        src_addr_base,
        dst_region_type,
        dst_region,
        dst_addr_base,
        ch,
    ):  # TODO: add more parameters
        assert src_region_type is Region.DRAM
        tile_shape = tile_attr["tile_shape"]
        if dst_region_type is Region.I_SRAM:
            x_stride = CTU_REDUCED_SIZE
        elif dst_region_type is Region.O_SRAM:
            x_stride = PSUM_CH_SIZE
        else:
            x_stride = 1

        x = ch
        y = tile_shape[1]
        z = tile_shape[0]
        len_xyz = XYZ(x, y, z)
        if src_region_type is Region.DRAM:
            src_stride = XY(x, x * call_node.args[0].checked_type.shape[2])
            dst_stride = XY(x_stride, x_stride * y)
        else:
            src_stride = XY(x_stride, x_stride * y)
            dst_stride = XY(x, x * call_node.checked_type.shape[2])

        ctu_dep_tag = 0
        for ctu_op in reversed(self.ctu_op_list):
            if (
                (
                    dst_region_type is Region.O_SRAM
                    and ctu_op.ofm_bank_start == dst_region[0]
                    and ctu_op.ofm_sram_base <= dst_addr_base
                    and ctu_op.ofm_sram_base + ctu_op.ofm_shape.h * ctu_op.ofm_shape.w
                    > dst_addr_base
                )
                or (
                    dst_region_type is Region.I_SRAM
                    and ctu_op.ifm_bank_start <= dst_region[0]
                    and ctu_op.ifm_bank_start + ctu_op.ifm_bank_span > dst_region[0]
                    and ctu_op.ifm_sram_base <= dst_addr_base // CTU_REDUCED_SIZE
                    and ctu_op.ifm_sram_base + ctu_op.ifm_shape.h * ctu_op.ifm_shape.w
                    > dst_addr_base // CTU_REDUCED_SIZE
                )
            ) and self.ctu_dep_tag - ctu_op.dep_tag.ctu_tag < 8:
                ctu_dep_tag = ctu_op.dep_tag.ctu_tag
                break

        ppu_dep_tag = 0
        for ppu_op in reversed(self.ppu_op_list):
            if (
                dst_region_type is Region.O_SRAM
                and (
                    ppu_op.ofm_bank_start == dst_region[0]
                    and ppu_op.ofm_sram_base <= dst_addr_base
                    and ppu_op.ofm_sram_base + ppu_op.ofm_shape.h * ppu_op.ofm_shape.w
                    > dst_addr_base
                    or ppu_op.ifm_bank_start == dst_region[0]
                    and ppu_op.ifm_sram_base <= dst_addr_base
                    and ppu_op.ifm_sram_base + ppu_op.ifm_shape.h * ppu_op.ifm_shape.w
                    > dst_addr_base
                )
                and self.ppu_dep_tag - ppu_op.dep_tag.ppu_tag < 8
            ):
                ppu_dep_tag = ppu_op.dep_tag.ppu_tag
                break

        st_dep_tag = 0
        for st_op in reversed(self.st_op_list):
            if (
                st_op.src_region_type == dst_region_type
                and st_op.src_region[0] == dst_region[0]
                and self.st_dep_tag - st_op.dep_tag.st_tag < 8
            ):
                st_dep_tag = st_op.dep_tag.st_tag
                break

        ld_dep_tag = 0
        if len(self.ld_op_list) > 0 and self.ld_op_list[-1] == self.macro_ops[-1]:
            self.ld_op_list[-1].dep_tag.ld_tag = 0
            ld_dep_tag = self.ld_dep_tag
            ctu_dep_tag = 0
            ppu_dep_tag = 0
            st_dep_tag = 0
        else:
            self.ld_dep_tag += 1
            ld_dep_tag = self.ld_dep_tag

        self.num_load_macro_ops += 1

        return LoadMacroOp(
            len_xyz=len_xyz,
            src_region_type=src_region_type,
            src_region=src_region,
            dst_region_type=dst_region_type,
            dst_region=dst_region,
            src_base=src_addr_base,
            dst_base=dst_addr_base,
            src_stride=src_stride,
            dst_stride=dst_stride,
            dep_tag=DependencyTag(ld_dep_tag, ctu_dep_tag, ppu_dep_tag, st_dep_tag),
        )

    def gen_store_macro_op(
        self,
        idx,
        call_node,
        tile_attr,
        src_region_type,
        src_region,
        src_addr_base,
        dst_region_type,
        dst_region,
        dst_addr_base,
        ch,
    ):  # TODO: add more parameters
        assert src_region_type is Region.I_SRAM or src_region_type is Region.O_SRAM
        tile_shape = tile_attr["o_tile_shape"]
        if src_region_type is Region.I_SRAM:
            x_stride = CTU_REDUCED_SIZE
        elif src_region_type is Region.O_SRAM:
            x_stride = PSUM_CH_SIZE
        else:
            x_stride = 1

        x = ch
        y = tile_shape[1]
        z = tile_shape[0]
        len_xyz = XYZ(x, y, z)
        if src_region_type is Region.DRAM:
            src_stride = XY(x, x * call_node.args[0].checked_type.shape[2])
            dst_stride = XY(x_stride, x_stride * y)
        else:
            src_stride = XY(x_stride, x_stride * y)
            dst_stride = XY(x, x * call_node.checked_type.shape[2])
        ctu_dep_tag = 0
        ppu_dep_tag = 0
        for ctu_op in reversed(self.ctu_op_list):
            if (
                ctu_op.ofm_bank_start == src_region[0]
                and ctu_op.ofm_sram_base <= src_addr_base
                and ctu_op.ofm_sram_base + ctu_op.ofm_shape.h * ctu_op.ofm_shape.w > src_addr_base
                and self.ctu_dep_tag - ctu_op.dep_tag.ctu_tag < 8
            ):
                ctu_dep_tag = ctu_op.dep_tag.ctu_tag
                break

        for ppu_op in reversed(self.ppu_op_list):
            if (
                ppu_op.ofm_bank_start == src_region[0]
                and ppu_op.ofm_sram_base <= src_addr_base
                and ppu_op.ofm_sram_base + ppu_op.ofm_shape.h * ppu_op.ofm_shape.w > src_addr_base
                and self.ppu_dep_tag - ppu_op.dep_tag.ppu_tag < 8
            ):
                ppu_dep_tag = ppu_op.dep_tag.ppu_tag
                break

        if len(self.st_op_list) > 0 and self.st_op_list[-1] == self.macro_ops[-1]:
            self.st_op_list[-1].dep_tag.st_tag = 0
            ctu_dep_tag = 0
            ppu_dep_tag = 0
        else:
            self.st_dep_tag += 1

        st_tag = self.st_dep_tag

        self.num_store_macro_ops += 1

        return StoreMacroOp(
            len_xyz=len_xyz,
            src_region_type=src_region_type,
            src_region=src_region,
            dst_region_type=dst_region_type,
            dst_region=dst_region,
            src_base=src_addr_base,
            dst_base=dst_addr_base,
            src_stride=src_stride,
            dst_stride=dst_stride,
            dep_tag=DependencyTag(0, ctu_dep_tag, ppu_dep_tag, st_tag),
        )

    def gen_bin_eltwise_macro_op(self, call_node, tile_attr):  # TODO: add more parameters
        ifm0 = call_node.args[BinEltwiseArgs.kIfm0Idx.value]
        ifm1 = call_node.args[BinEltwiseArgs.kIfm1Idx.value]
        bin_eltwise_op_type = BinEltwiseOpType[call_node.attrs.op_type]
        ifm_shape = HWC(
            tile_attr["tile_shape"][0],
            tile_attr["tile_shape"][1],
            tile_attr["tile_shape"][2],
        )  # HWC
        ifm_sram_base = self.cur_psum_allocation_map[ifm0][0]
        ifm_bank_start = self.cur_psum_buffer_idx * PSUM_BANK_GROUP_SIZE
        ifm_dtype = None
        if ifm0.checked_type.dtype == "int8":
            ifm_dtype = Dtype.INT8
        elif ifm0.checked_type.dtype == "uint8":
            ifm_dtype = Dtype.UINT8
        elif ifm0.checked_type.dtype == "int32":
            ifm_dtype = Dtype.INT32
        else:
            assert False, f"Unsupported ifm dtype: {ifm0.checked_type.dtype}"
        ifm_bank_span = 1 if ifm_dtype == Dtype.INT8 or ifm_dtype == Dtype.UINT8 else 4

        ifm1_sram_base = self.cur_psum_allocation_map[ifm1][0]
        ifm1_bank_start = self.cur_psum_buffer_idx * PSUM_BANK_GROUP_SIZE
        ifm1_dtype = None
        if ifm1.checked_type.dtype == "int8":
            ifm1_dtype = Dtype.INT8
        elif ifm1.checked_type.dtype == "uint8":
            ifm1_dtype = Dtype.UINT8
        elif ifm1.checked_type.dtype == "int32":
            ifm1_dtype = Dtype.INT32
        else:
            assert False, f"Unsupported ifm dtype: {ifm1.checked_type.dtype}"
        ifm1_bank_span = 1 if ifm1_dtype == Dtype.INT8 or ifm1_dtype == Dtype.UINT8 else 4

        ofm_shape = HWC(
            tile_attr["o_tile_shape"][0],
            tile_attr["o_tile_shape"][1],
            tile_attr["o_tile_shape"][2],
        )
        ofm_sram_base = ifm_sram_base
        ofm_bank_start = ifm_bank_start
        ofm_dtype = None
        if call_node.checked_type.dtype == "int8":
            ofm_dtype = Dtype.INT8
        elif call_node.checked_type.dtype == "uint8":
            ofm_dtype = Dtype.UINT8
        elif call_node.checked_type.dtype == "int32":
            ofm_dtype = Dtype.INT32
        else:
            assert False, f"Unsupported ofm dtype: {call_node.checked_type.dtype}"
        ofm_bank_span = 1 if ofm_dtype == Dtype.INT8 or ofm_dtype == Dtype.UINT8 else 4
        act_type = call_node.attrs.activation
        if hasattr(call_node.attrs, "activation"):
            if call_node.attrs.activation == "NONE":
                act_type = ActivationType.NONE
            elif call_node.attrs.activation == "CLIP":
                act_type = ActivationType.CLIP
            elif call_node.attrs.activation == "LUT":
                act_type = ActivationType.LUT
            else:
                assert False, f"Unsupported activation type: {call_node.attrs.activation}"

        broadcast_h = call_node.attrs.broadcast_h
        broadcast_w = call_node.attrs.broadcast_w

        ld_dep_tag = 0
        for ld_op in reversed(self.ld_op_list):
            if (
                ld_op.dst_region_type == Region.I_SRAM
                and (
                    (
                        ld_op.dst_region[0] == ifm_bank_start
                        and ifm_sram_base <= ld_op.dst_base
                        and ifm_sram_base + ifm_shape.h * ifm_shape.w > ld_op.dst_base
                    )
                    or (
                        ld_op.dst_region[0] == ifm1_bank_start
                        and ifm1_sram_base <= ld_op.dst_base
                        and ifm1_sram_base + ifm_shape.h * ifm_shape.w > ld_op.dst_base
                    )
                )
                and ld_op.dep_tag.ld_tag - self.ld_dep_tag < 8
            ):
                ld_dep_tag = ld_op.dep_tag.ld_tag
                break

        ctu_dep_tag = 0
        for ctu_op in reversed(self.ctu_op_list):
            if (
                ctu_op.ofm_bank_start == ofm_bank_start
                and ctu_op.ofm_sram_base <= ofm_sram_base
                and ctu_op.ofm_sram_base + ctu_op.ofm_shape.h * ctu_op.ofm_shape.w > ofm_sram_base
                or ctu_op.ofm_bank_start == ifm_bank_start
                and ctu_op.ofm_sram_base <= ifm_sram_base
                and ctu_op.ofm_sram_base + ctu_op.ofm_shape.h * ctu_op.ofm_shape.w > ifm_sram_base
                or ctu_op.ofm_bank_start == ifm1_bank_start
                and ctu_op.ofm_sram_base <= ifm1_sram_base
                and ctu_op.ofm_sram_base + ctu_op.ofm_shape.h * ctu_op.ofm_shape.w > ifm1_sram_base
                and self.ctu_dep_tag - ctu_op.dep_tag.ctu_tag < 8
            ):
                ctu_dep_tag = ctu_op.dep_tag.ctu_tag
                break

        st_dep_tag = 0
        for st_op in reversed(self.st_op_list):
            if (
                st_op.src_region[0] == ofm_bank_start
                and ofm_sram_base <= st_op.src_base
                and ofm_sram_base + ofm_shape.h * ofm_shape.w > st_op.src_base
                and st_op.dep_tag.st_tag - self.st_dep_tag < 8
            ):
                st_dep_tag = st_op.dep_tag.st_tag
                break

        self.ppu_dep_tag += 1
        ppu_dep_tag = self.ppu_dep_tag

        self.num_bin_eltwise_macro_ops += 1

        return BinEltwiseMacroOp(
            bin_eltwise_op_type=bin_eltwise_op_type,
            ofm_shape=ofm_shape,
            ifm_shape=ifm_shape,
            ifm_sram_base=ifm_sram_base,
            ifm_bank_start=ifm_bank_start,
            ifm_bank_span=ifm_bank_span,
            ifm_dtype=ifm_dtype,
            ifm1_sram_base=ifm1_sram_base,
            ifm1_bank_start=ifm1_bank_start,
            ifm1_bank_span=ifm1_bank_span,
            ofm_sram_base=ofm_sram_base,
            ofm_bank_start=ofm_bank_start,
            ofm_bank_span=ofm_bank_span,
            ofm_dtype=ofm_dtype,
            act_type=act_type,
            broadcast_h=broadcast_h,
            broadcast_w=broadcast_w,
            broadcast_c=False,
            dep_tag=DependencyTag(ld_dep_tag, ctu_dep_tag, ppu_dep_tag, st_dep_tag),
        )

    def gen_un_eltwise_macro_op(self, call_node, op_name, tile_attr):
        is_imm = not (op_name == "qnn.requantize" or call_node.attrs["op_type"] == "CLZ")
        un_eltwise_op_type = UnEltwiseOpType.REQ  # default
        if op_name == "qnn.requantize":
            pass  # requantize has dirrerent op name from bin_imm_eltwise
        elif call_node.attrs["op_type"] == "CLZ":
            un_eltwise_op_type = UnEltwiseOpType.CLZ
        elif call_node.attrs["op_type"] == "ADD":
            un_eltwise_op_type = UnEltwiseOpType.ADD_IMM
        elif call_node.attrs["op_type"] == "SUB":
            if call_node.attrs["is_ifm0"]:
                un_eltwise_op_type = UnEltwiseOpType.SUB_VAR_IMM
            else:
                un_eltwise_op_type = UnEltwiseOpType.SUB_IMM_VAR
        elif call_node.attrs["op_type"] == "SHR":
            if call_node.attrs["is_ifm0"]:
                un_eltwise_op_type = UnEltwiseOpType.SHR_VAR_IMM
            else:
                un_eltwise_op_type = UnEltwiseOpType.SHR_IMM_VAR
        elif call_node.attrs["op_type"] == "SHL":
            if call_node.attrs["is_ifm0"]:
                un_eltwise_op_type = UnEltwiseOpType.SHL_VAR_IMM
            else:
                un_eltwise_op_type = UnEltwiseOpType.SHL_IMM_VAR
        else:
            assert "Undefined unary elementwise op type"
        imm_high = None
        imm_low = None
        if is_imm:
            imm_high = 0xFFFF0000 & call_node.attrs["immediate"] >> 16
            imm_low = 0xFFFF & call_node.attrs["immediate"]

        ifm = call_node.args[ReqArgs.kIfmIdx.value]
        scale = (
            call_node.args[ReqArgs.kIfmScaleIdx.value].data.asnumpy()
            / call_node.args[ReqArgs.kOfmScaleIdx.value].data.asnumpy()
            if not is_imm
            else None
        )
        if tile_attr["scale_offset"] == -1:
            scale_base = -1
            scale_shift, scale_mantissa = get_fp_data(scale) if scale is not None else (0, 0)
        else:
            scale_base = self.const_addr_map[scale] + tile_attr["scale_offset"]
            scale_shift = 0
            scale_mantissa = 0

        ifm_shape = HWC(
            tile_attr["tile_shape"][0],
            tile_attr["tile_shape"][1],
            tile_attr["tile_shape"][2],
        )  # HWC
        ifm_sram_base = self.cur_psum_allocation_map[ifm][0]
        ifm_bank_start = self.cur_psum_buffer_idx * PSUM_BANK_GROUP_SIZE
        ifm_dtype = None
        ifm_zp = (
            int(call_node.args[ReqArgs.kIfmZpIdx.value].data.numpy())
            if un_eltwise_op_type == UnEltwiseOpType.REQ
            else 0
        )
        if ifm.checked_type.dtype == "int8":
            ifm_dtype = Dtype.INT8
        elif ifm.checked_type.dtype == "uint8":
            ifm_dtype = Dtype.UINT8
        else:
            assert False, f"Unsupported ifm dtype: {ifm.checked_type.dtype}"
        ifm_bank_span = 1 if ifm_dtype == Dtype.INT8 or ifm_dtype == Dtype.UINT8 else 4

        ofm_shape = HWC(
            tile_attr["o_tile_shape"][0],
            tile_attr["o_tile_shape"][1],
            tile_attr["o_tile_shape"][2],
        )
        ofm_sram_base = ifm_sram_base
        ofm_bank_start = ifm_bank_start
        ofm_dtype = None
        ofm_zp = (
            int(call_node.args[ReqArgs.kOfmZpIdx.value].data.numpy())
            if un_eltwise_op_type == UnEltwiseOpType.REQ
            else 0
        )

        if call_node.checked_type.dtype == "int8":
            ofm_dtype = Dtype.INT8
        elif call_node.checked_type.dtype == "uint8":
            ofm_dtype = Dtype.UINT8
        elif call_node.checked_type.dtype == "int32":
            ofm_dtype = Dtype.INT32
        else:
            assert False, f"Unsupported ofm dtype: {call_node.checked_type.dtype}"
        ofm_bank_span = 1 if ofm_dtype == Dtype.INT8 or ofm_dtype == Dtype.UINT8 else 4

        act_type = ActivationType.CLIP
        if hasattr(call_node.attrs, "activation"):
            if call_node.attrs.activation == "NONE":
                act_type = ActivationType.NONE
            elif call_node.attrs.activation == "CLIP":
                act_type = ActivationType.CLIP
            elif call_node.attrs.activation == "LUT":
                act_type = ActivationType.LUT
            else:
                assert False, f"Unsupported activation type: {call_node.attrs.activation}"

        req_en = False
        if tile_attr["to_dram"]:
            req_en = False
        else:
            req_en = True

        self.ppu_dep_tag += 1
        ppu_dep_tag = self.ppu_dep_tag
        ld_dep_tag = 0
        if self.ld_op_list[-1].dst_region_type == Region.O_SRAM:
            ld_dep_tag = self.ld_dep_tag

        ctu_dep_tag = 0
        for ctu_op in reversed(self.ctu_op_list):
            if (
                ctu_op.ofm_bank_start == ofm_bank_start
                and ctu_op.ofm_sram_base <= ofm_sram_base
                and ctu_op.ofm_sram_base + ctu_op.ofm_shape.h * ctu_op.ofm_shape.w > ofm_sram_base
                or ctu_op.ofm_bank_start == ofm_bank_start
                and ctu_op.ofm_sram_base <= ifm_sram_base
                and ctu_op.ofm_sram_base + ctu_op.ofm_shape.h * ctu_op.ofm_shape.w > ifm_sram_base
                and self.ppu_dep_tag - ctu_op.dep_tag.ppu_tag < 8
            ):
                ctu_dep_tag = ctu_op.dep_tag.ctu_tag
                break

        st_dep_tag = 0
        if len(self.st_op_list) > 0:
            for st_op in reversed(self.st_op_list):
                if (
                    st_op.src_region[0] == ofm_bank_start
                    and st_op.dep_tag.st_tag - self.st_dep_tag < 8
                ):
                    st_dep_tag = st_op.dep_tag.st_tag
                    break

        self.num_un_eltwise_macro_ops += 1

        return UnEltwiseMacroOp(
            un_eltwise_op_type=un_eltwise_op_type,
            imm_high=imm_high,
            imm_low=imm_low,
            ofm_shape=ofm_shape,
            ifm_shape=ifm_shape,
            ifm_sram_base=ifm_sram_base,
            ifm_bank_start=ifm_bank_start,
            ifm_bank_span=ifm_bank_span,
            ifm_dtype=ifm_dtype,
            ifm_zp=ifm_zp,
            ofm_sram_base=ofm_sram_base,
            ofm_bank_start=ofm_bank_start,
            ofm_bank_span=ofm_bank_span,
            scale_mantissa=scale_mantissa,
            scale_shift=scale_shift,
            scale_base=scale_base,
            ofm_dtype=ofm_dtype,
            ofm_zp=ofm_zp,
            act_type=act_type,
            req_en=req_en,
            dep_tag=DependencyTag(ld_dep_tag, ctu_dep_tag, ppu_dep_tag, st_dep_tag),
        )

    def gen_pool_macro_op(self, call_node, idx, is_last, tile_attr):  # TODO: add more parameters
        ifm = call_node.args[Pool2dArgs.kIfmIdx.value]

        ifm_shape = HWC(
            tile_attr["tile_shape"][0],
            tile_attr["tile_shape"][1],
            tile_attr["tile_shape"][2],
        )  # HWC
        ifm_sram_base = self.cur_psum_allocation_map[ifm][0]
        ifm_bank_start = self.cur_psum_buffer_idx * PSUM_BANK_GROUP_SIZE
        ifm_dtype = None
        if ifm.checked_type.dtype == "int8":
            ifm_dtype = Dtype.INT8
        elif ifm.checked_type.dtype == "uint8":
            ifm_dtype = Dtype.UINT8
        else:
            assert False, f"Unsupported ifm dtype: {ifm.checked_type.dtype}"
        ifm_bank_span = 1 if ifm_dtype == Dtype.INT8 or ifm_dtype == Dtype.UINT8 else 4
        ifm_zp = call_node.attrs.ifm_zp

        padding = TLBR(
            t=tile_attr["padding"][0],
            l=tile_attr["padding"][1],
            b=tile_attr["padding"][2],
            r=tile_attr["padding"][3],
        )
        kernel_shape = HW(
            call_node.attrs.pool_size[0],
            call_node.attrs.pool_size[1],
        )
        strides = HW(
            call_node.attrs.strides[0],
            call_node.attrs.strides[1],
        )
        window_info = WindowInfo(
            padding=padding,
            kernel_shape=kernel_shape,
            strides=strides,
        )

        ofm_shape = HWC(
            tile_attr["o_tile_shape"][0],
            tile_attr["o_tile_shape"][1],
            tile_attr["o_tile_shape"][2],
        )
        ofm_zp = call_node.attrs.ofm_zp
        ofm_sram_base = ifm_sram_base
        ofm_bank_start = ifm_bank_start
        ofm_dtype = None
        if call_node.checked_type.dtype == "int8":
            ofm_dtype = Dtype.INT8
        elif call_node.checked_type.dtype == "uint8":
            ofm_dtype = Dtype.UINT8
        else:
            assert False, f"Unsupported ofm dtype: {call_node.checked_type.dtype}"
        ofm_bank_span = 1 if ofm_dtype == Dtype.INT8 or ofm_dtype == Dtype.UINT8 else 4

        act_type = call_node.attrs.activation
        if call_node.attrs.activation == "NONE":
            act_type = ActivationType.NONE
        elif call_node.attrs.activation == "CLIP":
            act_type = ActivationType.CLIP
        else:
            assert False, f"Unsupported activation type: {call_node.attrs.activation}"

        pool_type = None
        if call_node.attrs.pool_type == "AVG":
            pool_type = PoolOpType.AVG_POOL
            scale_shift, scale_mantissa = get_fp_data(1.0 / (kernel_shape.h * kernel_shape.w))
        elif call_node.attrs.pool_type == "MAX":
            pool_type = PoolOpType.MAX_POOL
            (scale_shift, scale_mantissa) = (0, 0)
        else:
            assert False, f"Unsupported pool type: {call_node.attrs.pool_type}"
        scale_base = -1

        accu_en = idx != 0
        req_en = is_last

        self.ppu_dep_tag += 1
        ppu_dep_tag = self.ppu_dep_tag
        ld_dep_tag = 0
        if self.ld_op_list[-1].dst_region_type == Region.O_SRAM:
            ld_dep_tag = self.ld_dep_tag

        ctu_dep_tag = 0
        for ctu_op in reversed(self.ctu_op_list):
            if (
                ctu_op.ofm_bank_start == ofm_bank_start
                and ctu_op.ofm_sram_base <= ofm_sram_base
                and ctu_op.ofm_sram_base + ctu_op.ofm_shape.h * ctu_op.ofm_shape.w > ofm_sram_base
                or ctu_op.ofm_bank_start == ofm_bank_start
                and ctu_op.ofm_sram_base <= ifm_sram_base
                and ctu_op.ofm_sram_base + ctu_op.ofm_shape.h * ctu_op.ofm_shape.w > ifm_sram_base
                and self.ppu_dep_tag - ctu_op.dep_tag.ppu_tag < 8
            ):
                ctu_dep_tag = ctu_op.dep_tag.ctu_tag
                break

        st_dep_tag = 0
        if len(self.st_op_list) > 0:
            for st_op in reversed(self.st_op_list):
                if (
                    st_op.src_region[0] == ofm_bank_start
                    and st_op.dep_tag.st_tag - self.st_dep_tag < 8
                ):
                    st_dep_tag = st_op.dep_tag.st_tag
                    break
        self.num_pool_macro_ops += 1

        return PoolMacroOp(
            ofm_shape=ofm_shape,
            ifm_shape=ifm_shape,
            ifm_sram_base=ifm_sram_base,
            ifm_bank_start=ifm_bank_start,
            ifm_bank_span=ifm_bank_span,
            ifm_dtype=ifm_dtype,
            ifm_zp=ifm_zp,
            window_info=window_info,
            ofm_zp=ofm_zp,
            ofm_sram_base=ofm_sram_base,
            ofm_bank_start=ofm_bank_start,
            ofm_bank_span=ofm_bank_span,
            scale_base=scale_base,
            scale_mantissa=scale_mantissa,
            scale_shift=scale_shift,
            ofm_dtype=ofm_dtype,
            act_type=act_type,
            pool_op_type=pool_type,
            req_en=req_en,
            accu_en=accu_en,
            dep_tag=DependencyTag(ld_dep_tag, ctu_dep_tag, ppu_dep_tag, st_dep_tag),
        )

    def gen_reduce_macro_op(self, call_node, tile_attr):  # TODO: add more parameters
        if call_node.attrs["reduce_op"] == "SUM":
            reduce_op_type = ReduceOpType.SUM
        elif call_node.attrs["reduce_op"] == "MAX":
            reduce_op_type = ReduceOpType.MAX
        else:
            assert False, f"Unsupported reduce op type: {call_node.attrs['reduce_op']}"
        reduce_dim = HWC(call_node.attrs["reduce_h"], call_node.attrs["reduce_w"], False)
        ifm = call_node.args[ReqArgs.kIfmIdx.value]
        scale = call_node.args[ReqArgs.kIfmScaleIdx.value]
        if tile_attr["scale_offset"] == -1:
            scale_base = -1
            scale_shift, scale_mantissa = (
                get_fp_data(scale.data.asnumpy()) if scale is not None else (0, 0)
            )
        else:
            scale_base = self.const_addr_map[scale] + tile_attr["scale_offset"]
            scale_shift = 0
            scale_mantissa = 0

        ifm_shape = HWC(
            tile_attr["tile_shape"][0],
            tile_attr["tile_shape"][1],
            tile_attr["tile_shape"][2],
        )  # HWC
        ifm_sram_base = self.cur_psum_allocation_map[ifm][0]
        ifm_bank_start = self.cur_psum_buffer_idx * PSUM_BANK_GROUP_SIZE
        ifm_dtype = None
        if ifm.checked_type.dtype == "int8":
            ifm_dtype = Dtype.INT8
        elif ifm.checked_type.dtype == "uint8":
            ifm_dtype = Dtype.UINT8
        else:
            assert False, f"Unsupported ifm dtype: {ifm.checked_type.dtype}"
        ifm_bank_span = 1 if ifm_dtype == Dtype.INT8 or ifm_dtype == Dtype.UINT8 else 4

        ofm_shape = HWC(
            tile_attr["o_tile_shape"][0],
            tile_attr["o_tile_shape"][1],
            tile_attr["o_tile_shape"][2],
        )
        ofm_sram_base = ifm_sram_base
        ofm_bank_start = ifm_bank_start
        ofm_dtype = None

        if call_node.checked_type.dtype == "int8":
            ofm_dtype = Dtype.INT8
        elif call_node.checked_type.dtype == "uint8":
            ofm_dtype = Dtype.UINT8
        elif call_node.checked_type.dtype == "int32":
            ofm_dtype = Dtype.INT32
        else:
            assert False, f"Unsupported ofm dtype: {call_node.checked_type.dtype}"
        ofm_bank_span = 1 if ofm_dtype == Dtype.INT8 or ofm_dtype == Dtype.UINT8 else 4

        act_type = ActivationType.CLIP
        if hasattr(call_node.attrs, "activation"):
            if call_node.attrs.activation == "NONE":
                act_type = ActivationType.NONE
            elif call_node.attrs.activation == "CLIP":
                act_type = ActivationType.CLIP
            elif call_node.attrs.activation == "LUT":
                act_type = ActivationType.LUT
            else:
                assert False, f"Unsupported activation type: {call_node.attrs.activation}"

        req_en = False
        if tile_attr["to_dram"]:
            req_en = False
        else:
            req_en = True

        self.ppu_dep_tag += 1
        ppu_dep_tag = self.ppu_dep_tag
        ld_dep_tag = 0
        if self.ld_op_list[-1].dst_region_type == Region.O_SRAM:
            ld_dep_tag = self.ld_dep_tag

        ctu_dep_tag = 0
        for ctu_op in reversed(self.ctu_op_list):
            if (
                ctu_op.ofm_bank_start == ofm_bank_start
                and ctu_op.ofm_sram_base <= ofm_sram_base
                and ctu_op.ofm_sram_base + ctu_op.ofm_shape.h * ctu_op.ofm_shape.w > ofm_sram_base
                or ctu_op.ofm_bank_start == ofm_bank_start
                and ctu_op.ofm_sram_base <= ifm_sram_base
                and ctu_op.ofm_sram_base + ctu_op.ofm_shape.h * ctu_op.ofm_shape.w > ifm_sram_base
                and self.ppu_dep_tag - ctu_op.dep_tag.ppu_tag < 8
            ):
                ctu_dep_tag = ctu_op.dep_tag.ctu_tag
                break

        st_dep_tag = 0
        if len(self.st_op_list) > 0:
            for st_op in reversed(self.st_op_list):
                if (
                    st_op.src_region[0] == ofm_bank_start
                    and st_op.dep_tag.st_tag - self.st_dep_tag < 8
                ):
                    st_dep_tag = st_op.dep_tag.st_tag
                    break
        self.num_reduce_macro_ops += 1

        return ReduceMacroOp(
            reduce_op_type=reduce_op_type,
            reduce_dim=reduce_dim,
            ofm_shape=ofm_shape,
            ifm_shape=ifm_shape,
            ifm_sram_base=ifm_sram_base,
            ifm_bank_start=ifm_bank_start,
            ifm_bank_span=ifm_bank_span,
            ifm_dtype=ifm_dtype,
            ofm_sram_base=ofm_sram_base,
            ofm_bank_start=ofm_bank_start,
            ofm_bank_span=ofm_bank_span,
            scale_mantissa=scale_mantissa,
            scale_shift=scale_shift,
            scale_base=scale_base,
            ofm_dtype=ofm_dtype,
            act_type=act_type,
            req_en=req_en,
            dep_tag=DependencyTag(ld_dep_tag, ctu_dep_tag, ppu_dep_tag, st_dep_tag),
        )

    def __call__(self, mod):
        for gv, func in mod.functions.items():
            if is_novella_func(func):
                with open(f"{get_artifacts_dir()}/test_relay.log", "w") as f:
                    f.write(f"{gv}:\n")
                    f.write(func.astext())
                self.visit(func)
        self.macro_ops.append(ExtSyncMacroOp())  # Add sync macro op at the end
        with open(get_artifacts_dir() + "/barrier.txt", "w") as file:
            for i, barrier in enumerate(self.barrier_nodes):
                file.write(f"{i}: {barrier}\n")

        return self.macro_ops
