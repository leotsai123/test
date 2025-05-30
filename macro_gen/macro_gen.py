import math
from turtle import pos
from typing import List, Tuple, Dict, Union, Optional
from model_construct.node import Node, HWC
from macro_gen.isa_def import (
    MemType,
    Dtype,
    TLBR,
    HW,
    XY,
    XYZ,
    ActivationType,
    WindowInfo,
    LoadMacroOp,
    LoadMacroOp_,
    LoadMicroOp_,
    LoadCimMacroOp,
    ConvMacroOp,
    StoreMacroOp,
    SanityCheckOp,
    PositionMap
)
from tiling.tiling import dtype_to_bytes
from npu_emulator.utils import assign_bitwidth

WINDOW_OP = ['CONV2D_OP']
from model_construct.op_attrs import (
    CONV2D_OP
)

CIM_COL = 8
CIM_IC1 = 128
CIM_IC2 = 32
CIM_OC1 = 32
CIM_OC2 = 16
CIM_MAC = 64
CIM_SEG = 8
SEG_ELE = CIM_MAC

UNI_SRAM_SIZE = 16 * 1024 # 16KB
W_SRAM_SIZE = CIM_MAC * CIM_COL * CIM_SEG # 4KB

class MacroOpGen:
    def __init__(self, node_tile_attrs_map: List[Dict[str, Union[List, Tuple, int, bool]]], ch_per_group, mode:str):
        self.node_tile_attrs_map = node_tile_attrs_map
        self.mode=mode
        self.macro_ops = []
        self.num_macro_ops = 0
        self.conv_macro_ops = 0
        self.load_macro_ops = 0
        self.store_macro_ops = 0
        self.i_tile_src_base_addr=-1
        
        self.ifm_start=0
        self.wsram_start=0
        self.psum_start=0
        self.psum_len=0
        self.ofm_start=0
        self.ofm_len=0
        self.bias_start=0
        self.bias_len=0
        self.ch_per_group=ch_per_group

    def gen_macro_ops(self, callnode: Node, tile_attrs: Dict[str, Union[List, Tuple, int, bool]]):
        """
        Supposedly generate macro level operations
        """
        isram_cache_hit = (
            callnode.op == CONV2D_OP and tile_attrs["i_tile_offset"] == self.i_tile_src_base_addr
        )
        i_tile_shape = tile_attrs["i_tile_shape"]
        i_tile_h, i_tile_w= i_tile_shape[0], i_tile_shape[1]
        ifm_h, ifm_w, ifm_c = callnode.ifm_shape.h, callnode.ifm_shape.w, callnode.ifm_shape.c
        p_ifm_c = math.ceil(ifm_c/self.ch_per_group) * self.ch_per_group # ifm_c must be padded to multiples of self.ch_per_group before filling the ic segment 
            

        o_tile_shape = HWC(*tile_attrs["o_tile_shape"]) # * unpack tuple
        o_tile_h, o_tile_w, o_tile_c = o_tile_shape.h, o_tile_shape.w, o_tile_shape.c
        k_h, k_w = callnode.kernel_shape.h, callnode.kernel_shape.w
        
        if callnode.op != CONV2D_OP:
            raise ValueError("Not supported operations!!!")
            # spatial_size = CIM_OC
        else:
            reduce_size1 = CIM_IC1
            reduce_size2 = CIM_IC2
            spatial_size1 = CIM_OC1
            spatial_size2 = CIM_OC2
            # NOTE: align_mode flag is set to false for continuous memory access in vrf,
            # if the current ofm is for post process, data nust be aligned in vrf
            align_mode=False 
            overwrite=True

            reduce_groups1 = ifm_c // reduce_size1 if ifm_c >= reduce_size1 else 1

            if ifm_c >= reduce_size1:
                reduce_groups2 = reduce_size1 // reduce_size2
            elif ifm_c >= reduce_size2: # ifm_c can fill all ic segments
                reduce_groups2 = ifm_c // reduce_size2
            else:
                reduce_groups2 = 1 # ifm_c can only fill some of the ic segments
                reduce_size2 = p_ifm_c
            
        spatial_groups = spatial_size1 // spatial_size2

        # allocate ifm space
        ifm_data_byte = dtype_to_bytes(callnode.ifm_dtype)
        # self.ifm_start = 0 # TODO: add wrapped vrf feature
        self.ifm_len = i_tile_h * i_tile_w * p_ifm_c * ifm_data_byte
        # allocate psum space
        psum_data_byte = dtype_to_bytes(callnode.psum_dtype)
        self.psum_start =  self.ifm_start + self.ifm_len
        self.psum_len = o_tile_h * o_tile_w * o_tile_c * psum_data_byte
        # allocate ofm space
        ofm_data_byte = dtype_to_bytes(callnode.ofm_dtype)
        self.ofm_start = self.psum_start + self.psum_len if not overwrite else self.psum_start
        self.ofm_len = o_tile_h * o_tile_w * o_tile_c * ofm_data_byte
        # allocate bias space
        bias_data_byte = dtype_to_bytes(callnode.bias_dtype)
        self.bias_start = self.ofm_start + self.ofm_len if not overwrite else self.psum_start + self.psum_len
        self.bias_len = CIM_OC1 * bias_data_byte
        assert self.bias_start + self.bias_len < UNI_SRAM_SIZE, "Allocated psum + ifm + bias + ofm has exceeded the SRAM size!!!"

        ifm_dbyte=dtype_to_bytes(callnode.ifm_dtype)
        weight_dbyte=dtype_to_bytes(callnode.kernel_dtype)
        ofm_dbyte=dtype_to_bytes(callnode.ofm_dtype)
        psum_dbyte=dtype_to_bytes(callnode.psum_dtype)
        # split kernel size into 8 and 1
        k_size1 = 8
        k_size2 = 1
        reduced_group1=0
        ping=True
        # loads bias data into UNI_SRAMs
        if callnode.bias is not None:
            src_type = MemType.BIAS_MEM # data layout in DRAM is OC, 1D
            dst_type = MemType.UNI_SRAM
            dst_base_addr = self.bias_start
            src_base_addr = tile_attrs["bias_offset"]
            self.macro_ops.extend(
                self.gen_load_macro_op(
                    src_type=src_type,
                    dst_type=dst_type,
                    src_base_addr=src_base_addr,
                    dst_base_addr=dst_base_addr,
                    callnode=callnode,
                    mode=self.mode,
                    dbyte=bias_data_byte
                )
            )
        while reduced_group1 < reduce_groups1 and reduce_groups1 > 0:
            reduced_group2=0
            while reduced_group2 < reduce_groups2 and reduce_groups2 > 0 and k_h != 1 and k_w != 1:
                if tile_attrs["from_dram"] and not isram_cache_hit:
                    # loads i_tile into UNI_SRAM
                    src_type = MemType.DRAM # NHCWB8
                    dst_type = MemType.UNI_SRAM
                    src_base_addr = (
                        (reduced_group1 * reduce_size1 + reduced_group2 * reduce_size2) * ifm_w
                    ) * ifm_dbyte + tile_attrs["i_tile_offset"]
                    dst_base_addr =  (reduced_group1 * reduce_size1 + reduced_group2 * reduce_size2) * i_tile_w * ifm_dbyte

                    # i_tile block size is i_tile_h * CIM_IC2 * i_tile_w
                    self.macro_ops.extend(
                        self.gen_load_macro_op(
                            src_type=src_type,
                            dst_type=dst_type,
                            src_base_addr=src_base_addr,
                            dst_base_addr=dst_base_addr,
                            dst_stride=1,
                            tile_attrs=tile_attrs,
                            callnode=callnode,
                            k_size=k_size1,
                            mode=self.mode,
                            reduced_groups=(reduced_group1, reduced_group2),
                            reduce_size=(reduce_size1, reduce_size2),
                        )
                    )

                self.ifm_start = (reduced_group1 * reduce_size1 + reduced_group2 * reduce_size2) * i_tile_w # input channel offset
                is_ich_zero = (reduced_group1 == 0 and reduced_group2 == 0)
                
                
                # loads w_tile blocks into W_SRAM, block config is k=8, ic=8
                sp_group=0
                while sp_group < spatial_groups:
                    src_type = MemType.DRAM # data layout in DRAM is OCHWB8
                    dst_type = MemType.W_SRAM
                    # if current is ping load, assign wsram_start, else add offset
                    self.wsram_start = 0 if ping else W_SRAM_SIZE
                    dst_base_addr = self.wsram_start
                    src_base_addr = (
                        (sp_group * spatial_size2) * p_ifm_c * k_h * k_w +
                        (reduced_group1 * reduce_size1 + reduced_group2 * reduce_size2) * k_h * k_w * CIM_COL 
                    )* weight_dbyte + tile_attrs["weight_offset"]
                    self.macro_ops.extend(
                        self.gen_load_macro_op(
                            src_type=src_type,
                            dst_type=dst_type,
                            src_base_addr=src_base_addr,
                            dst_base_addr=dst_base_addr,
                            dst_stride=1,
                            tile_attrs=tile_attrs,
                            callnode=callnode,
                            k_size=k_size1,
                            mode=self.mode,
                            reduced_groups=(reduced_group1, reduced_group2),
                            reduce_size=(reduce_size1, reduce_size2),
                            ping=ping,
                            spatial_sizes=(spatial_size1, spatial_size2),
                            sp_group=sp_group,
                            dbyte=weight_dbyte,
                        )
                    )
                    is_last = False
                    
                    # gen conv macro op
                    cim_ic = p_ifm_c if p_ifm_c < CIM_IC2 else CIM_IC2
                    cim_oc = CIM_OC2
                    cim_sic = cim_ic // self.ch_per_group
                    cim_soc = cim_oc // self.ch_per_group
                    self.macro_ops.append(
                        self.gen_conv_macro_op(
                            is_ich_zero,
                            tile_attrs,
                            is_last,
                            callnode,
                            self.ifm_start,
                            self.psum_start,
                            self.ofm_start,
                            self.bias_start,
                            ping,
                            k_size1,
                            cim_ic,
                            cim_oc,
                            sp_group,
                            cim_sic,
                            cim_soc,
                            overwrite,
                            align_mode,
                        )
                    )
                    ping = not ping
                    self.wsram_start = 0 if ping else W_SRAM_SIZE
                    sp_group+=1
                reduced_group2+=1

            #TODO: add 1x1 ifm loading
            reduce_size1 = reduce_size1 if p_ifm_c >= reduce_size1 else (CIM_MAC // k_size2)
            if k_h == 1 and k_w == 1 and tile_attrs["from_dram"] and not isram_cache_hit:
                # loads i_tile into UNI_SRAM
                src_type = MemType.DRAM # NHCWB8
                dst_type = MemType.UNI_SRAM
                src_base_addr = (
                    (reduced_group1 * reduce_size1) * ifm_w
                )*ifm_dbyte  + tile_attrs["i_tile_offset"]
                dst_base_addr = (
                    (reduced_group1 * reduce_size1) * i_tile_w
                )*ifm_dbyte

                self.macro_ops.extend(
                    self.gen_load_macro_op(
                        src_type=src_type,
                        dst_type=dst_type,
                        src_base_addr=src_base_addr,
                        dst_base_addr=dst_base_addr,
                        dst_stride=1,
                        tile_attrs=tile_attrs,
                        callnode=callnode,
                        k_size=k_size1,
                        mode=self.mode,
                        reduced_groups=(reduced_group1, reduced_group2),
                        reduce_size=(reduce_size1, reduce_size2),
                    )
                )
                    
            # loads w_tile blocks into W_SRAM, block config is k=1, ic=64
            sp_group=0
            src_type = MemType.DRAM
            dst_type = MemType.W_SRAM
            self.wsram_start = 0 if ping else W_SRAM_SIZE
            dst_base_addr = self.wsram_start
            src_base_addr=(
                (reduced_group1 * reduce_size1) * k_h * k_w * CIM_COL +
                k_size1*CIM_COL)* weight_dbyte + tile_attrs["weight_offset"] if k_h == 3 and k_w ==3 else \
                ((reduced_group1 * reduce_size1) * k_h * k_w * CIM_COL)* weight_dbyte + tile_attrs["weight_offset"]
            self.macro_ops.extend(
                self.gen_load_macro_op(
                    src_type=src_type,
                    dst_type=dst_type,
                    src_base_addr=src_base_addr,
                    dst_base_addr=dst_base_addr,
                    dst_stride=1,
                    tile_attrs=tile_attrs,
                    callnode=callnode,
                    k_size=k_size2,
                    mode=self.mode,
                    reduced_groups=(reduced_group1, reduced_group2),
                    reduce_size=(reduce_size1, reduce_size2),
                    ping=ping,
                    spatial_sizes=(spatial_size1, spatial_size2),
                    sp_group=sp_group,
                    dbyte=weight_dbyte,
                )
            )
           
            is_last = reduced_group1 == reduce_groups1-1
            self.ifm_start = (reduced_group1 * reduce_size1) * i_tile_w # input channel offset
            is_ich_zero = (reduced_group1 == 0) if k_h == 1 and k_w == 1 else False
            cim_oc = CIM_OC1
            cim_ic = p_ifm_c if p_ifm_c < CIM_IC1 else CIM_IC1
            cim_sic = cim_ic // (8*self.ch_per_group) if cim_ic >= (8 * self.ch_per_group) else 1
            cim_soc = cim_oc //self.ch_per_group 
            # gen conv macro op
            self.macro_ops.append(
                self.gen_conv_macro_op(
                    is_ich_zero,
                    tile_attrs,
                    is_last,
                    callnode,
                    self.ifm_start,
                    self.psum_start,
                    self.ofm_start,
                    self.bias_start,
                    ping,
                    k_size2,
                    cim_ic,
                    cim_oc,
                    sp_group,
                    cim_sic,
                    cim_soc,
                    overwrite,
                    align_mode,
                )
            )
            reduced_group1+=1
            ping = not ping
        # Free psum allocation
        self.psum_len = 0
        self.i_tile_src_base_addr = tile_attrs["i_tile_offset"]

        # Store o_tile into external memory
        if tile_attrs["to_dram"]:
            src_type = MemType.UNI_SRAM
            dst_type = MemType.DRAM # NHCWB8
            ofm_h,ofm_w,ofm_c = callnode.ofm_shape.h, callnode.ofm_shape.w, callnode.ofm_shape.c
            idx_h = 0
            while idx_h < o_tile_h:
                oc_group=0
                while oc_group < o_tile_c // CIM_COL: # Different oc segments
                    if not overwrite and not align_mode:
                        dst = (
                            oc_group * CIM_COL * ofm_w +
                            idx_h * ofm_c * ofm_w
                        )* ofm_dbyte + tile_attrs["o_tile_offset"]
                        src = (
                            idx_h * o_tile_c * o_tile_w +
                            oc_group * o_tile_w * CIM_COL
                        )* ofm_dbyte + self.ofm_start
                        src_len = o_tile_w * CIM_COL * ofm_dbyte
                        self.macro_ops.append(
                            self.gen_store_macro_op(
                                src_type,
                                src_len,
                                src,
                                dst_type,
                                dst
                            )
                        )
                    elif overwrite and not align_mode:
                        for idx_w in range(o_tile_w):
                            dst = (
                                idx_h * ofm_c * ofm_w +
                                oc_group * CIM_COL * ofm_w +
                                idx_w * CIM_COL
                            )* ofm_dbyte + tile_attrs["o_tile_offset"]
                            src = (
                                idx_h * o_tile_c * o_tile_w +
                                oc_group * o_tile_w * CIM_COL +
                                idx_w * CIM_COL
                            )* psum_dbyte + self.ofm_start
                            src_len = CIM_COL * ofm_dbyte
                            self.macro_ops.append(
                                self.gen_store_macro_op(
                                    src_type,
                                    src_len,
                                    src,
                                    dst_type,
                                    dst
                                )
                            )
                    oc_group+=1
                idx_h+=1
                
        # Sanity check Op
        self.macro_ops.append(self.gen_sanity_check_op(tile_attrs))
        # Since the result has been written out, the ifm_start address should be reset
        self.ifm_start=0
    
    def gen_load_macro_op(
        self, 
        src_type: MemType, 
        dst_type: MemType,
        mode: str,
        src_base_addr: int = 0,
        dst_base_addr: int = 0,
        dst_stride: int = 1,
        tile_attrs=None,
        dbyte: int = 1, 
        reduce_size: Tuple = (1,), 
        callnode: Node = None, 
        k_size: int = 8, 
        reduced_groups: Tuple = (), 
        ping: bool = False, 
        spatial_sizes: Tuple = (), 
        sp_group: int = 0,
    ) -> List[Union[LoadMacroOp,LoadMacroOp_,LoadCimMacroOp]]:
        
        ops = []
        if dst_type == MemType.UNI_SRAM and src_type == MemType.BIAS_MEM:
            src_len = CIM_OC1 * dbyte
            op = LoadMacroOp(
                src_region_type=src_type,
                src_len=src_len,
                src=src_base_addr,
                dst_region_type=dst_type,
                dst=dst_base_addr,
            )
            ops.append(op)
            if mode == "low":
                # macro op
                x_len = CIM_OC1
                y_len = z_len = 0
                xyz_len=XYZ(x_len,y_len,z_len)
                src_x_stride = src_y_stride = 0
                dst_x_stride = dst_y_stride = 0
                src_base_addr=src_base_addr
                dst_base_addr=dst_base_addr
                src_stride=XY(src_x_stride,src_y_stride)
                dst_stride=XY(dst_x_stride,dst_y_stride)
                ops.append(
                    LoadMacroOp_(
                        xyz_len=xyz_len,
                        src_base_addr=src_base_addr,
                        dst_base_addr=dst_base_addr,
                        src_type=src_type,
                        dst_type=dst_type,
                        src_stride=src_stride,
                        dst_stride=dst_stride,
                    )
                )
                # micro op
                seg_num = 1
                v_stride = 1
                seg_stride = 0
                seg_len = src_len * dbyte
                op = LoadMicroOp_(
                    src_type=src_type,
                    dst_type=dst_type,
                    src_base_addr=src_base_addr,
                    dst_base_addr=dst_base_addr,
                    seg_num=seg_num,
                    v_stride=v_stride,
                    seg_stride=seg_stride,
                    seg_len=seg_len
                )
                ops.append(op)
            return ops
        
        ifm_w, ifm_c = callnode.ifm_shape.w, callnode.ifm_shape.c
        k_h, k_w = callnode.kernel_shape.h, callnode.kernel_shape.w
        p_ifm_c = math.ceil(ifm_c/self.ch_per_group) * self.ch_per_group
        i_tile_shape=HWC(*tile_attrs["i_tile_shape"])
        i_tile_h, i_tile_w = i_tile_shape.h, i_tile_shape.w
        reduced_group1, reduced_group2 = reduced_groups[0], reduced_groups[1]
        reduce_size1,reduce_size2 = reduce_size[0], reduce_size[1]
        if dst_type == MemType.UNI_SRAM:
            src=src_base_addr
            idx_h=0
            if k_h == 1 and k_w == 1:
                while idx_h < i_tile_h:
                    src += idx_h * p_ifm_c * ifm_w * dbyte
                    dst = (
                        idx_h * p_ifm_c * i_tile_w +
                        (reduced_group1 * reduce_size1) * i_tile_w
                    ) * dbyte
                    reduced_ch1=0
                    reduced_ch1_bound = math.ceil(reduce_size1 / (CIM_MAC // k_size))
                    while reduced_ch1 < reduced_ch1_bound: # loads for different ic segments
                        reduced_ch2=0
                        reduced_ch2_bound = (CIM_MAC // k_size // self.ch_per_group) if p_ifm_c >= (CIM_MAC // k_size) \
                        else (p_ifm_c // self.ch_per_group)
                        while reduced_ch2 < reduced_ch2_bound: # every ic segment has contiguous (CIM_MAC // k_size2) elements
                            src_len = i_tile_w * self.ch_per_group * dbyte
                            op = LoadMacroOp(
                                src_region_type=src_type,
                                src_len=src_len,
                                src=src,
                                dst_region_type=dst_type,
                                dst=dst,
                            )
                            ops.append(op)

                            src += ifm_w*self.ch_per_group* dbyte
                            dst += src_len
                            reduced_ch2+=1
                        reduced_ch1+=1
                    idx_h+=1
                    src=src_base_addr
            else:
                while idx_h < i_tile_h:
                    src += idx_h * p_ifm_c * ifm_w * dbyte
                    dst = (
                        idx_h * p_ifm_c * i_tile_w +
                        (reduced_group1 * reduce_size1 + reduced_group2 * reduce_size2) * i_tile_w
                    ) * dbyte
                    reduced_ch=0
                    while reduced_ch < reduce_size2 // self.ch_per_group: # loads for different ic segments
                        src_len = i_tile_w * self.ch_per_group * dbyte
                        op = LoadMacroOp(
                            src_region_type=src_type,
                            src_len=src_len,
                            src=src,
                            dst_region_type=dst_type,
                            dst=dst,
                        )
                        ops.append(op)
                        
                        dst += src_len
                        src += ifm_w*self.ch_per_group* dbyte
                        reduced_ch+=1
                    idx_h+=1
                    src = src_base_addr

            if mode == "low":
                x_len = i_tile_w
                y_len = reduced_ch2_bound * reduced_ch1_bound if k_h == 1 and k_w == 1 else (reduce_size2 // self.ch_per_group)
                z_len = i_tile_h
                xyz_len=XYZ(x_len,y_len,z_len)
                src_x_stride = ifm_w * self.ch_per_group * dbyte
                src_y_stride = p_ifm_c * ifm_w * dbyte
                dst_x_stride=x_len
                dst_y_stride=p_ifm_c * i_tile_w * dbyte
                src_base_addr=src_base_addr
                dst_base_addr=dst_base_addr
                src_stride=XY(src_x_stride,src_y_stride)
                dst_stride=XY(dst_x_stride,dst_y_stride)
                continuous_flag = (p_ifm_c > CIM_IC1 and k_h == 1 and k_w == 1) or (p_ifm_c > CIM_IC2 and k_h != 1 and k_w != 1)
                # macro op
                ops.append(
                    LoadMacroOp_(
                        xyz_len=xyz_len,
                        src_base_addr=src_base_addr,
                        dst_base_addr=dst_base_addr,
                        src_type=src_type,
                        dst_type=dst_type,
                        src_stride=src_stride,
                        dst_stride=dst_stride,
                        continuous_flag=continuous_flag,
                    )
                )
                # micro op
                if continuous_flag:
                   for z in range(z_len):
                        seg_num = y_len
                        v_stride = 0
                        seg_stride = src_stride.x
                        seg_len = x_len * self.ch_per_group * dbyte
                        op = LoadMicroOp_(
                            src_type=src_type,
                            dst_type=dst_type,
                            src_base_addr=src_base_addr,
                            dst_base_addr=dst_base_addr,
                            seg_num=seg_num,
                            v_stride=v_stride,
                            seg_stride=seg_stride,
                            seg_len=seg_len
                        )
                        ops.append(op)

                        src_base_addr += src_stride.y
                        dst_base_addr += dst_stride.y
                else:
                    seg_num = y_len * z_len
                    v_stride = 0
                    seg_stride = src_stride.x
                    seg_len = x_len * self.ch_per_group * dbyte
                    op = LoadMicroOp_(
                        src_type=src_type,
                        dst_type=dst_type,
                        src_base_addr=src_base_addr,
                        dst_base_addr=dst_base_addr,
                        seg_num=seg_num,
                        v_stride=v_stride,
                        seg_stride=seg_stride,
                        seg_len=seg_len
                    )
                    ops.append(op)

            self.load_macro_ops +=1
            return ops
        
        if dst_type == MemType.W_SRAM:
            spatial_size1, spatial_size2 = spatial_sizes[0], spatial_sizes[1]
            # if current is ping load, assign wsram_start, else add offset
            self.wsram_start = 0 if ping else W_SRAM_SIZE
            dst = self.wsram_start
            
            oc_group=0
            if k_size != 1:  # k=8, ic=8 weight loading
                while oc_group < spatial_size2 // CIM_COL: # loads for different oc segments
                    reduced_ch1=0
                    src_org=src_base_addr+(
                        (oc_group * CIM_COL) * p_ifm_c * k_h * k_w
                    )* dbyte
                    while reduced_ch1 < reduce_size2 // (CIM_MAC // k_size): # loads for different ic segments
                        # each ic segments has (CIM_MAC // k_size1) ic and k_size1 kernel, in this case, it is 8,8
                        src=src_org+(
                            reduced_ch1 * (CIM_MAC // k_size) * k_h * k_w * CIM_COL
                        )* dbyte
                        reduced_ch2=0
                        reduced_ch2_bound = (CIM_MAC // k_size) if p_ifm_c >= (CIM_MAC // k_size) else p_ifm_c
                        while reduced_ch2 < reduced_ch2_bound: # iterate through each inner ic (CIM_MAC // k_size1)
                            src_len = k_size * CIM_COL * dbyte # every inner ic has contiguous k_size1*B8 elements
                            op = LoadMacroOp(
                                src_region_type=src_type,
                                src_len=src_len,
                                src=src,
                                dst_region_type=dst_type,
                                dst=dst,
                            )
                            ops.append(op)
                            src += k_h * k_w * CIM_COL * dbyte
                            dst += src_len
                            reduced_ch2+=1
                        reduced_ch1+=1
                    oc_group+=1
            else: # k=1, ic=64 weight loading
                while oc_group < spatial_size1 // CIM_COL: # loads for different oc segments
                    reduced_ch1=0
                    src_org = src_base_addr + (
                        (oc_group * CIM_COL) * p_ifm_c * k_h * k_w
                    )* dbyte
                    while reduced_ch1 < reduce_size1 // (CIM_MAC // k_size): # loads for different ic segments
                        src = src_org+(
                            reduced_ch1 * (CIM_MAC // k_size) * k_h * k_w * CIM_COL
                        )* dbyte
                        reduced_ch2=0
                        reduced_ch2_bound = (CIM_MAC // k_size) if p_ifm_c >= (CIM_MAC // k_size) else p_ifm_c
                        while reduced_ch2 < reduced_ch2_bound: # every ic segment has contiguous (CIM_MAC // k_size2) elements
                            len = (CIM_MAC // k_size) if p_ifm_c >= (CIM_MAC // k_size) else p_ifm_c
                            src_len = (len * self.ch_per_group * dbyte) if k_h == 1 and k_w == 1 else \
                                (k_size * CIM_COL * dbyte)
                            
                            op=LoadMacroOp(
                                src_type,
                                src_len,
                                src,
                                dst_type,
                                dst
                            )
                            ops.append(op)
                            
                            src+= (k_h*k_w*CIM_COL)* dbyte
                            dst += src_len
                            reduced_ch2+= (CIM_MAC // k_size) if k_h == 1 and k_w == 1 else 1
                        reduced_ch1+=1
                    oc_group+=1
            
            if mode == "low":
                # Compute spatial output channel (soc)
                if k_size != 1:
                    soc = spatial_size2 // CIM_COL
                else:
                    soc = spatial_size1 // CIM_COL

                # Stride for output channels (in bytes)
                oc_stride = CIM_COL * p_ifm_c * k_h * k_w * dbyte

                # Compute spatial input channel (sic)
                if k_size != 1:
                    sic = reduce_size2 // (CIM_MAC // k_size)
                else:
                    sic = reduce_size1 // (CIM_MAC // k_size)

                # Stride for input channels (in bytes)
                ic_stride = (CIM_MAC // k_size) * k_h * k_w * CIM_COL * dbyte

                # Effective input channels (depending on padding size)
                eic = min(p_ifm_c, CIM_MAC // k_size)

                # Kernel stride (in bytes)
                k_stride = k_h * k_w * CIM_COL * dbyte

                # Compute source length for DMA
                if k_size != 1:
                    src_len = k_size * CIM_COL * dbyte
                else:
                    # For k_size == 1, kernel shape may be pointwise
                    len_eic = min(p_ifm_c, CIM_MAC // k_size)
                    if k_h == 1 and k_w == 1:
                        src_len = len_eic * CIM_COL * dbyte
                    else:
                        src_len = k_size * CIM_COL * dbyte

                ops.append(
                    LoadCimMacroOp(
                        soc=soc,
                        oc_stride=oc_stride,
                        sic=sic,
                        ic_stride=ic_stride,
                        eic=eic,
                        k_stride=k_stride,
                        src_base_addr=src_base_addr,
                        src_len=src_len,
                        ping=ping,
                        k_size=k_size,
                        kernel_hw=(k_h,k_w),
                    )
                )
            
            
            return ops
    
    def gen_store_macro_op(
        self,
        src_region_type, 
        src_len, 
        src, 
        dst_region_type,
        dst
    ):
        self.store_macro_ops +=1
        return StoreMacroOp(
            src_region_type=src_region_type,
            src_len=src_len,
            src=src,
            dst_region_type=dst_region_type,
            dst=dst
        )
    
    def gen_conv_macro_op(
            self,
            is_ich_zero,
            tile_attrs,
            is_last: bool,
            callnode: Node,
            ifm_start,
            psum_start,
            ofm_start,
            bias_start,
            ping,
            k_size,
            cim_ic,
            cim_oc,
            sp_group,
            cim_sic,
            cim_soc,
            overwrite,
            align_mode,
    ):
        self.conv_macro_ops += 1

        # Currently only support scale that is the same across all channels
        if tile_attrs["scale_offset"] == -1:
            scale_base = -1
            scale_shift, scale_mantissa = callnode.scale_shift, callnode.scale_mantissa
        else:
            raise ValueError("Scale mode isn't supported!!!")
            # scale_base = tile_attrs["scale_offset"]
            # scale_shift = 0
            # scale_mantissa = 0

        ifm_c=tile_attrs["i_tile_shape"][2]
        p_ifm_c = math.ceil(ifm_c/self.ch_per_group) * self.ch_per_group

        ifm_shape = HWC(
            tile_attrs["i_tile_shape"][0],
            tile_attrs["i_tile_shape"][1],
            p_ifm_c,
        )  # HWC

        if callnode.ifm_dtype == "int8":
            ifm_dtype = Dtype.INT8
        elif callnode.ifm_dtype == "uint8":
            ifm_dtype = Dtype.UINT8
        else:
            raise ValueError("Not supported ifm_dtype!!!")
        ifm_zp = callnode.ifm_zp

        ifm_sram_base = ifm_start
        
        padding = TLBR(
            t=tile_attrs["padding"][0],
            l=tile_attrs["padding"][1],
            b=tile_attrs["padding"][2],
            r=tile_attrs["padding"][3],
        )
        kernel_shape = HW(
           callnode.kernel_shape.h,
           callnode.kernel_shape.w
        )
        strides = HW(
            callnode.stride.x,
            callnode.stride.y
        )
        # dilation = HW(
        #     callnode.dilation.x,
        #     callnode.dilation.y,
        # )

        if k_size == 8:
            posmap = PositionMap(
                0, # 1
                0, # 2
                1, # 3
                0, # 4
                0, # 5
                1, # 6
                0, # 7
                1, # 8
            )
        else:
            posmap = PositionMap(
                1, # 1
                1, # 2
                1, # 3
                1, # 4
                1, # 5
                1, # 6
                1, # 7
                1, # 8
            )

        window_info = WindowInfo(
            padding=padding,
            kernel_shape=kernel_shape,
            strides=strides,
            # dilation=dilation,
        )

        if callnode.kernel_dtype == "int8":
            kernel_dtype = Dtype.INT8
        elif callnode.kernel_dtype == "uint8":
            kernel_dtype = Dtype.UINT8


        ofm_shape = HWC(
            tile_attrs["o_tile_shape"][0],
            tile_attrs["o_tile_shape"][1],
            tile_attrs["o_tile_shape"][2],
        )  # HWC
        ofm_zp=callnode.ofm_zp
        if callnode.ofm_dtype == "int8":
            ofm_dtype = Dtype.INT8
        elif callnode.ofm_dtype == "uint8":
            ofm_dtype = Dtype.UINT8
        else:
            raise ValueError("Not supported ofm_dtype!!!")
        
        # TODO: Basicly the psum will be released right after being read, so it can be overwritten with the ofm result
        # CAUTION: Still needs to check with Wayne for this logic
        psum_sram_base=psum_start

        act_type = callnode.activation
        if callnode.activation is None:
            act_type = ActivationType.NONE
            act_min=0
            act_max=0
        elif callnode.activation == "ReLU":
            act_type = ActivationType.RELU
            act_min=0
            act_max=2**(assign_bitwidth(ofm_dtype))-1
        elif callnode.activation == "Clamp":
            act_type = ActivationType.CLAMP
            act_min=-(2**(assign_bitwidth(ofm_dtype)-1)-1)
            act_max=2**(assign_bitwidth(ofm_dtype)-1)-1
        else:
            assert False, f"Unsupported activation type: {callnode.activation}"

        
        req_en = True if is_last else False             # requantization enable
        accu_en = True if not is_ich_zero else False    # accumulation enable
        bias_en = True if is_ich_zero else False        # bias add enable

        bias_sram_base = bias_start
        ofm_sram_base = ofm_start

        i_tile_coord = tile_attrs["i_tile_coord"] # debug usage
        o_tile_coord = tile_attrs["o_tile_coord"] # debug usage

        oc_group1 = tile_attrs["oc_group"] # debug usage -> grouped as CIM_OC1: 32
        oc_group2 = oc_group1 * (CIM_OC1 // CIM_OC2) + sp_group # debug usage -> grouped as CIM_OC2: 16
        # debug usage -> sends oc_group1 when k_size is 1, oc_group2 when k_size is 2
        oc_group = oc_group1 if k_size == 1 else oc_group2
        return ConvMacroOp(
            ofm_shape=ofm_shape,
            ifm_shape=ifm_shape,
            ifm_sram_base=ifm_sram_base,
            ifm_dtype=ifm_dtype,
            ifm_zp=ifm_zp,
            window_info=window_info,
            posmap=posmap,
            kernel_dtype=kernel_dtype,
            bias_sram_base=bias_sram_base,
            scale_mantissa=scale_mantissa,
            scale_shift=scale_shift,
            scale_base=scale_base,
            ofm_sram_base=ofm_sram_base,
            psum_sram_base=psum_sram_base,
            ofm_dtype=ofm_dtype,
            ofm_zp=ofm_zp,
            act_type=act_type,
            act_max=act_max,
            act_min=act_min,
            accu_en=accu_en,
            req_en=req_en,
            bias_en=bias_en,
            ping=ping,
            k_size=k_size,
            sp_group=sp_group,
            cim_sic=cim_sic,
            cim_soc=cim_soc,
            overwrite=overwrite,
            align_mode=align_mode,
            cim_ic=cim_ic, # debug usage
            cim_oc=cim_oc, # debug usage
            i_tile_coord=i_tile_coord, # debug usage
            oc_group=oc_group, # debug usage
            o_tile_coord=o_tile_coord # debug usage
        )

    def gen_sanity_check_op(
        self,
        tile_attrs
    ):
        return SanityCheckOp(tile_attrs)
        
    def __call__(self, callnode: Node):
        for tile_attrs in self.node_tile_attrs_map:
            self.gen_macro_ops(callnode, tile_attrs)
        return self.macro_ops