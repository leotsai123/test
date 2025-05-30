from re import S
from model_construct.node import Node, XY
from model_construct.layout import to_nhcwb8, to_ochwb8
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Union, Optional
from macro_gen.isa_def import LoadMacroOp, MemType, ConvMacroOp, StoreMacroOp, SanityCheckOp, ActivationType
from npu_emulator.utils import dtype_to_bytes, Shape, assign_torch_dtype, assign_np_dtype
from model_construct.node import HWC
import numpy as np
import math
import os



CIM_COL=8
CIM_MAC=64
SEG=8
W_SRAM_SIZE = CIM_COL * CIM_MAC * SEG # 4 KB


CIM_IC1 = 128
CIM_IC2 = 32
CIM_OC1 = 32
CIM_OC2 = 16

UNI_SRAM_SIZE = 16 * 1024 # 16KB
DRAM_SIZE = 1024 * 1024 * 10 # 10MB

class EmulateBuilder:
    def __init__(self, macro_ops: List[Union[LoadMacroOp,ConvMacroOp,StoreMacroOp,SanityCheckOp]], callnode: Node, ch_per_group: int, layer_idx: int):
        self.macro_ops = macro_ops
        self.callnode = callnode
        self.ch_per_group = ch_per_group
        self.ifm_extern = self.flatten_fm(callnode.ifm_data, self.ch_per_group, self.callnode.ifm_dtype)
        self.weight_extern = self.flatten_kernel(callnode.kernel_data, self.ch_per_group, self.callnode.kernel_dtype)
        self.bias_extern = self.flatten_bias(callnode.bias, self.ch_per_group, self.callnode.bias_dtype)
        self.ofm_gold = self.callnode.req_data
        self.ofm_extern = np.zeros(DRAM_SIZE, dtype=np.int8)
        self.unisram_onchip = np.zeros(UNI_SRAM_SIZE, dtype=np.int8)
        self.wsram_onchip = np.zeros(W_SRAM_SIZE*2, dtype=np.int8) # ping-pong weight sram
        self.ofm_sram_base=None
        self.psum_sram_base=None
        self.dumpy_ifm_sram_base=-1
        self.layer_idx=layer_idx
        self.ifm_block=0
        self.req_en_flag=0

    def reset(self):
        self.unisram_onchip = np.zeros(UNI_SRAM_SIZE, dtype=np.int8)
        self.wsram_onchip = np.zeros(W_SRAM_SIZE*2, dtype=np.int8) # ping-pong weight sram
        self.ofm_extern = np.zeros(DRAM_SIZE, dtype=np.int8)
        self.ofm_sram_base=None
        self.psum_sram_base=None
        self.dumpy_ifm_sram_base=-1
        self.ifm_block=0
        self.req_en_flag=0
    
    def flatten_fm(self, tensor: torch.tensor, ch_per_group: int, dtype: Union[str, MemType]) -> np.array:
        """
        Parameters
        ----------
            input tensor shape is NCHW, ch_per_group, dtype

        Returns
        -------
            Return flattened NHCWB8 tensor
        """
        tensor_dtype = to_nhcwb8(tensor, ch_per_group).to(assign_torch_dtype(dtype)).contiguous()

        # Convert to raw bytes (numpy → int8 view)
        np_bytes = tensor_dtype.cpu().numpy().view(np.int8)

        return np_bytes
    
    def flatten_kernel(self, tensor: torch.tensor, ch_per_group: int, dtype: Union[str, MemType]) -> np.array:
        """
        Parameters
        ----------
            kernel tensor shape is OCHW, ch_per_group, dtype

        Returns
        -------
            Return flattened OCHWB8 byte array
        """
        tensor_dtype = to_ochwb8(tensor, ch_per_group).to(assign_torch_dtype(dtype)).contiguous()

        # Convert to raw bytes (numpy → int8 view)
        np_bytes = tensor_dtype.cpu().numpy().view(np.int8).flatten()

        return np_bytes
    
    def flatten_bias(self, tensor: torch.Tensor, ch_per_group: int, dtype: Union[str, MemType]) -> np.array:
        """
        Convert float32 bias tensor to flattened int8 byte array (OB8 padded)

        Parameters
        ----------
        tensor : torch.Tensor
            Bias tensor of shape (O,), dtype=torch.float32
        ch_per_group : int
            OB8 alignment requirement (typically 8)

        Returns
        -------
        torch.Tensor
            Flattened bias bytes as np.int8 array of shape (padded_O * 4,)
        """
        import math

        assert tensor.dtype == torch.float32, "Expected float32 bias tensor"

        o = tensor.shape[0]
        padded_o = math.ceil(o / ch_per_group) * ch_per_group

        # Pad to OB8
        if padded_o != o:
            pad_len = padded_o - o
            pad_tensor = torch.zeros(pad_len, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, pad_tensor], dim=0)

        # Convert to int32 (4-byte representation)
        tensor_dtype = tensor.to(assign_torch_dtype(dtype)).contiguous()

        # Convert to raw bytes (numpy → int8 view)
        np_bytes = tensor_dtype.cpu().numpy().view(np.int8).flatten()

        return np_bytes

    def load_tile(
            self, 
            src: int, 
            src_len: int,
            dst: int,
            dst_type: MemType,
            src_type: MemType
        ):

        """
        Emulate wrapper behavior, when the ifm_c is not a multiple of 8, it should pad the input channel, 
        filled the bus with zeros so that each data being transfered is 64 bits 

        Parameters
        ----------
            Based on the tile src(offset) and length to load corresponding data on-chip

        Returns
        -------
            No return values, update on-chip sram content
        """
        if src_type == MemType.BIAS_MEM:
            self.unisram_onchip[dst:dst+src_len] = self.bias_extern[src:src+src_len]
        elif dst_type == MemType.UNI_SRAM:
            self.unisram_onchip[dst:dst+src_len] = self.ifm_extern[src:src+src_len]
        elif dst_type == MemType.W_SRAM:
            self.wsram_onchip[dst:dst+src_len] = self.weight_extern[src:src+src_len]

    def store_tile(
            self, 
            src: int, 
            src_len: int,
            dst: int,
            dst_type: MemType.DRAM
        ):

        """
        Parameters
        ----------
            Based on the tile src(offset) and length to store corresponding data off-chip

        Returns
        -------
            No return values, update off-chip RAM content
        """

        if dst_type == MemType.DRAM:
            self.ofm_extern[dst:dst+src_len] = self.unisram_onchip[src:src+src_len]
    
    def mac_op(
            self,
            ofm_shape,
            ifm_shape,
            ifm_sram_base,
            window_info,
            accu_en,
            req_en,
            ofm_sram_base,
            psum_sram_base,
            bias_sram_base,
            bias_en,
            ping,
            k_size,
            scale_mantissa,
            scale_shift,
            cim_ic,
            cim_oc,
            ifm_dtype,
            weight_dtype,
            psum_dtype,
            ofm_dtype,
            i_tile_coord,
            oc_group,
            sp_group,
            o_tile_coord,
            act_type,
            act_max,
            act_min
        ):
        """
        Parameters
        ----------
            on-chip i_tile and w_tile. i_tile is in 1D NHCWB8, w_tile is in 1D OCHWB8

        Returns
        -------
            No return, store the convolution results in 1D NHCWB8
        """

        build_dir = "../build/"
        dump_dir = os.path.join(build_dir, 'dump', 'layer'+str(self.layer_idx))
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        # dump the ifm data in dram
        if self.dumpy_ifm_sram_base == -1:
            ifm_extern_np = self.ifm_extern
            npy_path = os.path.join(dump_dir, f"layer{self.layer_idx}.npy")
            np.save(npy_path,ifm_extern_np)

        k_h, k_w = self.callnode.kernel_shape.h, self.callnode.kernel_shape.w
        # dump the ifm data in vrf
        if (self.dumpy_ifm_sram_base != ifm_sram_base or self.req_en_flag):
            if k_h == 3 and k_w == 3 and k_size == 1:
                pass
            else:
                self.dumpy_ifm_sram_base = ifm_sram_base
                npy_path = os.path.join(dump_dir, f"block{self.ifm_block}.npy")
                ifm_onchip_np = self.unisram_onchip[0:self.psum_sram_base]
                np.save(npy_path,ifm_onchip_np)
                self.ifm_block+=1

        self.req_en_flag=req_en

        ifm_dbyte=dtype_to_bytes(ifm_dtype)
        weight_dbyte=dtype_to_bytes(weight_dtype)
        bias_dbyte=psum_dbyte=dtype_to_bytes(psum_dtype)

        i_tile_coord = Shape(i_tile_coord)
        o_tile_coord = Shape(o_tile_coord)
        ifm_h, ifm_w = ifm_shape.h, ifm_shape.w
        
        ifm_c = self.callnode.ifm_shape.c
        p_ifm_c = math.ceil(ifm_c/self.ch_per_group) * self.ch_per_group
        ifm_tensor = torch.tensor([], dtype=assign_torch_dtype(ifm_dtype))

        unisram_onchip = torch.from_numpy(self.unisram_onchip)

        CIM_IC = cim_ic
        CIM_OC = cim_oc
        
        for idx_h in range(ifm_h):
            ifm_tensor = torch.cat((ifm_tensor,
                                    unisram_onchip[(idx_h*p_ifm_c*ifm_w*ifm_dbyte + ifm_sram_base):
                                                    (idx_h*p_ifm_c*ifm_w*ifm_dbyte + ifm_sram_base) + (ifm_w*CIM_IC*ifm_dbyte)]))
        

        # 1D -> (H, C//8, W, B8, byte)
        ifm_tensor = ifm_tensor.view(ifm_h, CIM_IC // self.ch_per_group, ifm_w, self.ch_per_group, ifm_dbyte)
        # Permute to (C//8, B8, H, W, byte)
        ifm_tensor = ifm_tensor.permute(1, 3, 0, 2, 4).contiguous()
        # Back to CHW
        ifm_tensor = ifm_tensor.contiguous().view(
            ifm_tensor.shape[0] * ifm_tensor.shape[1], 
            ifm_tensor.shape[2],
            ifm_tensor.shape[3]*ifm_tensor.shape[4]
        )

        # (1,C,H,W) -> ready for pytorch conv2d function
        ifm_tensor = ifm_tensor.unsqueeze(0)
        ifm_tensor_np = ifm_tensor[0].numpy()
        # Check whether the on-chip ifm is as same as the external one
        ichg = int(ifm_sram_base / ifm_w / CIM_IC)
        # external ifm (1,C,H,W), C must be padded
        if p_ifm_c != ifm_c:
            pad_channels = p_ifm_c - ifm_c
            tensor = self.callnode.ifm_data
            pad_tensor = torch.zeros((tensor.shape[0], pad_channels, self.callnode.ifm_shape.h, self.callnode.ifm_shape.w), dtype=tensor.dtype, device=tensor.device)
            ifm_data = torch.cat([tensor, pad_tensor], dim=1)
        else: ifm_data = self.callnode.ifm_data
        
        ifm_gold = ifm_data[:,
                            ichg*CIM_IC:ichg*CIM_IC+CIM_IC,
                            i_tile_coord.h:i_tile_coord.h+ifm_h,
                            i_tile_coord.w:i_tile_coord.w+ifm_w]
        
        # NOTE: ifm is of torch.ifm_dtype, and ifm_gold is torch.float32

        # Convert ifm_gold to match ifm_tensor dtype
        ifm_gold_ = ifm_gold.to(assign_torch_dtype(ifm_dtype))
        assert torch.equal(ifm_gold_, ifm_tensor), "onchip ifm is incorrect!!!"

        # ifm padding
        padding = window_info.padding
        # Compatible for pytorch padding
        pad = (padding.l, padding.r, padding.t, padding.b)
        # Padded ifm with shape of (1,C,pH,pW)
        ifm_tensor = F.pad(ifm_tensor, pad, mode='constant', value=0)

        w_len = k_size * CIM_IC * CIM_OC * weight_dbyte # TODO: CIM_OC needs judgement
        wsram_onchip = torch.from_numpy(self.wsram_onchip)
        weight = wsram_onchip[0:w_len] if ping else wsram_onchip[W_SRAM_SIZE:W_SRAM_SIZE+w_len]
        # 1D -> (O//8,C,k_size,8,byte)
        weight = weight.view(CIM_OC // CIM_COL, CIM_IC, k_size, CIM_COL, weight_dbyte)
        # Permute to (O//8, B8, C, k_size, weight_dbyte)
        weight = weight.permute(0, 3, 1, 2, 4).contiguous()
        # Back to OCK
        weight = weight.contiguous().view(
            weight.shape[0] * weight.shape[1], 
            weight.shape[2], 
            weight.shape[3] * weight.shape[4]
        )

        if p_ifm_c != ifm_c:
            pad_channels = p_ifm_c - ifm_c
            tensor = self.callnode.kernel_data
            pad_tensor = torch.zeros((self.callnode.kernel_shape.o, pad_channels, self.callnode.kernel_shape.h, self.callnode.kernel_shape.w), dtype=tensor.dtype, device=tensor.device)
            kernel_data = torch.cat([tensor, pad_tensor], dim=1)
        else: kernel_data = self.callnode.kernel_data

        # external weight (O,C,H,W)
        weight_gold = kernel_data[
            oc_group*CIM_OC:oc_group*CIM_OC+CIM_OC,
            ichg*CIM_IC:ichg*CIM_IC+CIM_IC,
            :,
            :
        ]

        # obtain the masked kernel to perform torch conv2d operation
        if k_h == 1 and k_w == 1:
            weight_stripped = weight_gold.view(weight_gold.shape[0],weight_gold.shape[1],-1)
            weight_tensor = weight_gold
        else:
            if k_size == 8:
                mask = torch.ones_like(weight_gold)
                mask[:, :, 2, 2] = 0 # Mask out the last element (bottom-right corner)
                weight_tensor = weight_gold * mask
            else:
                mask = torch.zeros(weight_gold.shape)
                mask[:, :, 2, 2] = 1 # retain only the bottom-right element
                weight_tensor = weight_gold * mask

            # stripped the kernel element regarding to the k_size
            indices_to_keep = [0,1,2,3,4,5,6,7] if k_size == 8 else [8]
            weight_gold = weight_gold.view(weight_gold.shape[0],weight_gold.shape[1],-1)
            weight_stripped = weight_gold[:, :, indices_to_keep] 

        # NOTE: weight is of torch.weight_dtype, and weight_stripped is torch.float32
        assert torch.equal(weight_stripped, weight), "onchip weight is incorrect!!!"

        strides = window_info.strides
        # data type sanity check
        weight_tensor=weight_tensor.to(assign_torch_dtype(weight_dtype))
        ifm_tensor=ifm_tensor.to(assign_torch_dtype(ifm_dtype))
        # Should output a tensor of shape (1,OC,H,W) dtype=torch.float32
        conv_result = F.conv2d(
            input=ifm_tensor.to(torch.float32),
            weight=weight_tensor.to(torch.float32),
            stride=strides.h,
        )
        conv_result.to(assign_torch_dtype(psum_dtype))
        if bias_en:
            bias_dtype=psum_dtype
            bias_gold = self.callnode.bias[oc_group*CIM_OC:oc_group*CIM_OC+CIM_OC]
            pivot=bias_sram_base+(sp_group)*CIM_OC*bias_dbyte if k_size == 8 else bias_sram_base
            bias_bytes = self.unisram_onchip[pivot:pivot+CIM_OC*bias_dbyte]
            bias_int32_np = bias_bytes.view(np.int32)  # shape: (CIM_OC,)
            bias_tensor = torch.from_numpy(bias_int32_np).to(dtype=assign_torch_dtype(bias_dtype), device='cuda' if torch.cuda.is_available() else 'cpu')
            assert torch.equal(bias_gold, bias_tensor), "onchip bias is incorrect!!!"
            conv_result+=bias_tensor.view(1,-1,1,1)
        
        # conv_result_flatten is a np byte array
        conv_result_flatten = self.flatten_fm(conv_result, CIM_COL, psum_dtype)
        accu_len=conv_result_flatten.shape[0]
        assert accu_len == ofm_shape.h * ofm_shape.w * CIM_OC * psum_dbyte, "Shape of conv result isn't correct!!!"

        # if idx_oc is even, meaning we are now dealing with the first CIM_OC2 group
        # if idx_oc is odd, meaning we are now dealing with the second CIM_OC2 group
        # idx_oc = oc_group % 2

        conv_result_unflatten = conv_result_flatten.reshape(ofm_shape.h, -1)
        if bias_en:
            self.psum_sram_base=psum_sram_base
            if k_size == 8:
                for idx_h in range(ofm_shape.h):
                    pivot = (
                        idx_h * CIM_OC1 * ofm_shape.w + 
                        sp_group * CIM_OC2 * ofm_shape.w
                    )* psum_dbyte + psum_sram_base
                    if pivot+CIM_OC2*ofm_shape.w*psum_dbyte > UNI_SRAM_SIZE:
                        raise ValueError(
                            f"k_size={k_size}, psum_sram_base={psum_sram_base}, idx_h={idx_h}, CIM_OC1={CIM_OC1}, ofm_shape.w={ofm_shape.w}, sp_group={sp_group}, pivot={pivot}, access len={CIM_OC2*ofm_shape.w*psum_dbyte}"
                        )
                    self.unisram_onchip[pivot:pivot+CIM_OC2*ofm_shape.w*psum_dbyte]=conv_result_unflatten[idx_h,:]
            else: 
                self.unisram_onchip[psum_sram_base:psum_sram_base+accu_len]=conv_result_flatten
                if psum_sram_base+accu_len > UNI_SRAM_SIZE:
                    raise ValueError(
                        f"k_size={k_size}, psum_sram_base={psum_sram_base}, access len={accu_len}"
                    )
        if accu_en:
            assert self.psum_sram_base == psum_sram_base, "Psum sram base address is incorrect!!!"
            if k_size == 8:
                for idx_h in range(ofm_shape.h):
                    pivot = (
                        idx_h * CIM_OC1 * ofm_shape.w + 
                        sp_group * CIM_OC2 * ofm_shape.w
                    )* psum_dbyte + psum_sram_base
                    if pivot+CIM_OC2*ofm_shape.w*psum_dbyte > UNI_SRAM_SIZE:
                        raise ValueError(
                            f"k_size={k_size}, psum_sram_base={psum_sram_base}, idx_h={idx_h}, CIM_OC1={CIM_OC1}, ofm_shape.w={ofm_shape.w}, sp_group={sp_group}, pivot={pivot}, access len={CIM_OC2*ofm_shape.w*psum_dbyte}"
                        )
                    unisram_onchip_bytes = self.unisram_onchip[pivot:pivot+CIM_OC2*ofm_shape.w*psum_dbyte]
                    unisram_onchip_int32 = unisram_onchip_bytes.view(np.int32)
                    conv_result_unflatten_bytes = conv_result_unflatten[idx_h,:]
                    conv_result_unflatten_int32 = conv_result_unflatten_bytes.view(np.int32)
                    result_int32 = conv_result_unflatten_int32 + unisram_onchip_int32
                    result_bytes = result_int32.view(np.int8)
                    self.unisram_onchip[pivot:pivot+CIM_OC2*ofm_shape.w*psum_dbyte]=result_bytes
            else: 
                unisram_onchip_bytes = self.unisram_onchip[psum_sram_base:psum_sram_base+accu_len]
                unisram_onchip_int32 = unisram_onchip_bytes.view(np.int32)
                conv_result_flatten_bytes = conv_result_flatten
                conv_result_flatten_int32 = conv_result_flatten_bytes.view(np.int32)
                result_int32 = conv_result_flatten_int32 + unisram_onchip_int32
                result_bytes = result_int32.view(np.int8)
                self.unisram_onchip[psum_sram_base:psum_sram_base+accu_len] = result_bytes
                if psum_sram_base+accu_len > UNI_SRAM_SIZE:
                    raise ValueError(
                        f"k_size={k_size}, psum_sram_base={psum_sram_base}, access len={accu_len}"
                    )
        if req_en:
            # Check with the convolution result first
            ofm_h, ofm_w = ofm_shape.h, ofm_shape.w
            ofm_result=self.callnode.ofm_data[
                :,
                oc_group*CIM_OC:oc_group*CIM_OC+CIM_OC,
                o_tile_coord.h:o_tile_coord.h+ofm_h,
                o_tile_coord.w:o_tile_coord.w+ofm_w
            ]
            ofm_result_flatten = self.flatten_fm(ofm_result, CIM_COL, psum_dtype)
            unisram_onchip = self.unisram_onchip[psum_sram_base:psum_sram_base+accu_len]

            assert (ofm_result_flatten==unisram_onchip).all(), "Convolution result stored in UNI_SRAM is incorrect!!!"

            # Check with the requantized result
            scale_mantissa = int(scale_mantissa)
            scale_shift = int(scale_shift)
            assert int(self.callnode.scale_mantissa) == scale_mantissa and int(self.callnode.scale_shift) == scale_shift, f"scaling factor is incorret!!! scale_mantissa = {scale_mantissa}, scale_shift = {scale_shift}. The correct scaling factor is scale_mantissa = {self.callnode.scale_mantissa}, scale_shift = {self.callnode.scale_shift}"
            # rounding = 1 << (scale_shift - 1)
            rounding=0
            unisram_onchip_bytes = self.unisram_onchip[psum_sram_base:psum_sram_base+accu_len]
            unisram_onchip_int32 = unisram_onchip_bytes.view(np.int32)

            req_result_int32 = (unisram_onchip_int32 * scale_mantissa + rounding) >> scale_shift
            if act_type == ActivationType.CLAMP or act_type == ActivationType.RELU:
                req_result_clipped = np.clip(req_result_int32, act_min, act_max)
            else: req_result_clipped = req_result_int32
            req_result_ofm_dtype = req_result_clipped.astype(assign_np_dtype(ofm_dtype))
            req_len = len(req_result_ofm_dtype)
            self.unisram_onchip[ofm_sram_base:ofm_sram_base+req_len] = req_result_ofm_dtype

            ofm_h, ofm_w = ofm_shape.h, ofm_shape.w
            ofm_result=self.callnode.req_data[
                :,
                oc_group*CIM_OC:oc_group*CIM_OC+CIM_OC,
                o_tile_coord.h:o_tile_coord.h+ofm_h,
                o_tile_coord.w:o_tile_coord.w+ofm_w
            ]
            ofm_result_flatten = self.flatten_fm(ofm_result, CIM_COL, ofm_dtype)
            unisram_onchip = self.unisram_onchip[ofm_sram_base:ofm_sram_base+req_len]
            

            assert (ofm_result_flatten==unisram_onchip).all(), "Requantization result stored in UNI_SRAM is incorrect!!!"

    def sanity_check(self, tile_attrs):
        """
        Extract content in ofm_extern based on tile_attrs to check with the content of ofm_gold
        """

        o_tile_shape = tile_attrs["o_tile_shape"]
        o_tile_coord = Shape(tile_attrs["o_tile_coord"])
        oc_group = tile_attrs["oc_group"]
        o_tile_shape = HWC(*o_tile_shape)
        o_tile_h, o_tile_w, o_tile_c = o_tile_shape.h, o_tile_shape.w, o_tile_shape.c
        ofm_h, ofm_w, ofm_c = self.callnode.ofm_shape.h, self.callnode.ofm_shape.w, self.callnode.ofm_shape.c
        ofm_dtype=self.callnode.ofm_dtype
        ofm_dbyte=dtype_to_bytes(ofm_dtype)
        
        ofm_extract = torch.zeros(o_tile_h*o_tile_w*o_tile_c, dtype=assign_torch_dtype(ofm_dtype))
        o_tile_offset = tile_attrs["o_tile_offset"]

        b8_idx=0
        for h_idx in range(o_tile_h):
            for oc_idx in range(o_tile_c // self.ch_per_group):
                for w_idx in range(o_tile_w):
                    addr = (
                        h_idx * ofm_c * ofm_w +
                        oc_idx * ofm_w * self.ch_per_group +
                        w_idx * self.ch_per_group
                    ) *ofm_dbyte + o_tile_offset
                    ofm_extract[b8_idx:b8_idx+self.ch_per_group*ofm_dbyte] = torch.from_numpy(self.ofm_extern[addr:addr+self.ch_per_group*ofm_dbyte].copy())
                    b8_idx+=self.ch_per_group*ofm_dbyte
                    
        # ofm_extern is 1D HCWB8,ofm_dbyte
        ofm_extract = ofm_extract.view(o_tile_h, o_tile_c // self.ch_per_group, o_tile_w, self.ch_per_group)
        # Permute to (C//8, B8, H, W) ofm_dbyte
        ofm_extract = ofm_extract.permute(1,3,0,2).contiguous()
        # Back to CHW ofm_dbyte
        ofm_extract = ofm_extract.contiguous().view(
            ofm_extract.shape[0] * ofm_extract.shape[1], 
            ofm_extract.shape[2],
            ofm_extract.shape[3]
        )

        # ofm_gold is CHW ofm_dbyte
        ofm_gold = self.ofm_gold.squeeze(0)
        ofm_gold = ofm_gold[
            oc_group*o_tile_c:oc_group*o_tile_c+o_tile_c,
            o_tile_coord.h:o_tile_coord.h+o_tile_h,
            o_tile_coord.w:o_tile_coord.w+o_tile_w,
        ]

        assert torch.equal(ofm_extract, ofm_gold), "Store result isn't correct!!!"


    def __call__(self):
        idx=0
        for macro_op in self.macro_ops:
            if isinstance(macro_op, LoadMacroOp):
                self.load_tile(
                    src=macro_op.src,
                    src_len=macro_op.src_len,
                    dst=macro_op.dst,
                    dst_type=macro_op.dst_region_type,
                    src_type=macro_op.src_region_type
                )
            if isinstance(macro_op, ConvMacroOp):
                self.mac_op(
                    ofm_shape=macro_op.ofm_shape,
                    ifm_shape=macro_op.ifm_shape,
                    ifm_sram_base=macro_op.ifm_sram_base,
                    window_info=macro_op.window_info,
                    ofm_sram_base=macro_op.ofm_sram_base,
                    psum_sram_base=macro_op.psum_sram_base,
                    bias_sram_base=macro_op.bias_sram_base,
                    accu_en=macro_op.accu_en,
                    req_en=macro_op.req_en,
                    bias_en=macro_op.bias_en,
                    scale_mantissa=macro_op.scale_mantissa,
                    scale_shift=macro_op.scale_shift,
                    i_tile_coord=macro_op.i_tile_coord,
                    oc_group=macro_op.oc_group,
                    sp_group=macro_op.sp_group,
                    o_tile_coord=macro_op.o_tile_coord,
                    ping=macro_op.ping,
                    k_size=macro_op.k_size,
                    cim_ic=macro_op.cim_ic,
                    cim_oc=macro_op.cim_oc,
                    ifm_dtype=macro_op.ifm_dtype,
                    weight_dtype=macro_op.kernel_dtype,
                    psum_dtype=macro_op.psum_dtype,
                    ofm_dtype=macro_op.ofm_dtype,
                    act_type=macro_op.act_type,
                    act_max=macro_op.act_max,
                    act_min=macro_op.act_min,
                )
            if isinstance(macro_op, StoreMacroOp):
                self.store_tile(
                    src=macro_op.src,
                    src_len=macro_op.src_len,
                    dst=macro_op.dst,
                    dst_type=macro_op.dst_region_type
                )
            if isinstance(macro_op, SanityCheckOp):
                # print(f"Tile {idx+1} sanity checking")
                self.sanity_check(tile_attrs=macro_op.tile_attrs)
                idx+=1