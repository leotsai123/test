from re import S
from model_construct.node import Node, XY
from model_construct.layout import to_nhcwb8, to_ochwb8
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Union, Optional
from macro_gen.isa_def import ActivationType, LoadMacroOp, MemType, ConvMacroOp, StoreMacroOp, SanityCheckOp, WindowInfo, ScaleModeType
from npu_emulator.utils import dtype_to_bytes, Shape, assign_torch_dtype, assign_np_dtype, pack_int8_to_int64
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

class EmulateValidator:
    def __init__(self, macro_ops: List[Union[LoadMacroOp,ConvMacroOp,StoreMacroOp,SanityCheckOp]], callnode: Node, ch_per_group: int, layer_idx: int):
        self.macro_ops = macro_ops
        self.callnode = callnode
        self.ch_per_group = ch_per_group
        self.ifm_extern = EmulateValidator.flatten_fm(callnode.ifm_data, self.ch_per_group, self.callnode.ifm_dtype)
        self.weight_extern = EmulateValidator.flatten_kernel(callnode.kernel_data, self.ch_per_group, self.callnode.kernel_dtype)
        self.bias_extern = EmulateValidator.flatten_bias(callnode.bias, self.ch_per_group, self.callnode.bias_dtype)
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
        self.ifm=None
        self.weight=None
        self.ofm=None
        self.bias=None
        self.req_idx=0
        self.dump_dir=None
        self.mac_result_queue: Dict[int,np.array] = {}
        self.psum_result_queue: Dict[int,np.array] = {}
        self.psum_fetch_queue: Dict[int,np.array] = {}
    
    @staticmethod
    def flatten_fm(tensor: torch.tensor, ch_per_group: int, dtype: Union[str, MemType]) -> np.array:
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
    
    @staticmethod
    def flatten_kernel(tensor: torch.tensor, ch_per_group: int, dtype: Union[str, MemType]) -> np.array:
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
    
    @staticmethod
    def flatten_bias(tensor: torch.Tensor, ch_per_group: int, dtype: Union[str, MemType]) -> np.array:
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
            bias_sram_base,
            ofm_sram_base,
            psum_sram_base,
            bias_en,
            ping,
            k_size,
            scale_mantissa,
            scale_shift,
            cim_ic,
            cim_oc,
            cim_sic,
            cim_soc,
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
            act_min,
            align_mode,
            overwrite,
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
        self.dump_dir = os.path.join(build_dir, 'dump', 'layer'+str(self.layer_idx))
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        k_h, k_w = self.callnode.kernel_shape.h, self.callnode.kernel_shape.w
        # # dump the ifm data in dram
        # if self.dumpy_ifm_sram_base == -1:
        #     ifm_extern_np = self.ifm_extern
        #     npy_path = os.path.join(dump_dir, f"layer{self.layer_idx}.npy")
        #     np.save(npy_path,ifm_extern_np)

        # # dump the ifm data in vrf
        # if (self.dumpy_ifm_sram_base != ifm_sram_base or self.req_en_flag):
        #     if k_h == 3 and k_w == 3 and k_size == 1:
        #         pass
        #     else:
        #         self.dumpy_ifm_sram_base = ifm_sram_base
        #         npy_path = os.path.join(dump_dir, f"block{self.ifm_block}.npy")
        #         ifm_onchip_np = self.unisram_onchip[0:self.psum_sram_base]
        #         np.save(npy_path,ifm_onchip_np)
        #         self.ifm_block+=1

        self.req_en_flag=req_en

        ifm_dbyte=dtype_to_bytes(ifm_dtype)
        weight_dbyte=dtype_to_bytes(weight_dtype)
        bias_dbyte=psum_dbyte=dtype_to_bytes(psum_dtype)
        ofm_dbyte=dtype_to_bytes(ofm_dtype)

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
                                                    (idx_h*p_ifm_c*ifm_w*ifm_dbyte + ifm_sram_base) + (ifm_w*CIM_IC*ifm_dbyte)].clone().to(assign_torch_dtype(ifm_dtype))))
        

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
        ifm_gold__np=ifm_gold_[0].numpy()
        ifm_tensor_np=ifm_tensor[0].numpy()
        equal = (ifm_gold__np==ifm_tensor_np)
        assert torch.equal(ifm_gold_, ifm_tensor), "onchip ifm is incorrect!!!"

        # ifm padding
        padding = window_info.padding
        # Compatible for pytorch padding
        pad = (padding.l, padding.r, padding.t, padding.b)
        # Padded ifm with shape of (1,C,pH,pW)
        ifm_tensor = F.pad(ifm_tensor, pad, mode='constant', value=0)
        ifm_tensor_np = ifm_tensor[0].numpy()
        self.ifm = ifm_tensor

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
        self.weight=weight.numpy()
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
        # Should output a tensor of shape (1,OC,H,W) dtype=torch.float32
        if k_size == 8:
            ic_span = self.ch_per_group
        else:
            if cim_ic < CIM_MAC: ic_span = cim_ic
            else: ic_span = CIM_MAC

        for icg in range(cim_sic):
            ifm_tensor_partial=ifm_tensor[:,icg*ic_span:icg*ic_span+ic_span,:,:]
            weight_tensor_partial=weight_tensor[:,icg*ic_span:icg*ic_span+ic_span,:,:]

            conv_result = F.conv2d(
                input=ifm_tensor_partial.to(torch.float32),
                weight=weight_tensor_partial.to(torch.float32),
                stride=strides.h,
            )
            conv_result.to(assign_torch_dtype(psum_dtype))
            self.mac_result_queue[icg]=conv_result[0].numpy().copy()

            if bias_en:
                bias_dtype=psum_dtype
                bias_gold = self.callnode.bias[oc_group*CIM_OC:oc_group*CIM_OC+CIM_OC]
                pivot=bias_sram_base+(sp_group)*CIM_OC*bias_dbyte if k_size == 8 else bias_sram_base
                bias_bytes = self.unisram_onchip[pivot:pivot+CIM_OC*bias_dbyte]
                bias_int32_np = bias_bytes.view(np.int32)  # shape: (CIM_OC,)
                self.bias=bias_int32_np
                bias_tensor = torch.from_numpy(bias_int32_np).to(dtype=assign_torch_dtype(bias_dtype), device='cuda' if torch.cuda.is_available() else 'cpu')
                assert torch.equal(bias_gold, bias_tensor), "onchip bias is incorrect!!!"
                if icg==0: conv_result+=bias_tensor.view(1,-1,1,1)
        
                # conv_result_flatten is a np byte array
                conv_result_flatten = EmulateValidator.flatten_fm(conv_result, CIM_COL, psum_dtype)
                accu_len=conv_result_flatten.shape[0]
                assert accu_len == ofm_shape.h * ofm_shape.w * CIM_OC * psum_dbyte, "Shape of conv result isn't correct!!!"

                conv_result_unflatten = conv_result_flatten.reshape(ofm_shape.h, -1)
        
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
                        if icg==0: self.unisram_onchip[pivot:pivot+CIM_OC2*ofm_shape.w*psum_dbyte]=conv_result_unflatten[idx_h,:]
                        else:
                            unisram_onchip_bytes = self.unisram_onchip[pivot:pivot+CIM_OC2*ofm_shape.w*psum_dbyte]
                            unisram_onchip_int32 = unisram_onchip_bytes.view(np.int32)
                            conv_result_unflatten_bytes = conv_result_unflatten[idx_h,:]
                            conv_result_unflatten_int32 = conv_result_unflatten_bytes.view(np.int32)
                            result_int32 = conv_result_unflatten_int32 + unisram_onchip_int32
                            result_bytes = result_int32.view(np.int8)
                            self.unisram_onchip[pivot:pivot+CIM_OC2*ofm_shape.w*psum_dbyte]=result_bytes
                else: 
                    if icg==0: self.unisram_onchip[psum_sram_base:psum_sram_base+accu_len]=conv_result_flatten
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
                # Reshape psum value into 3D array for easier debug usage
                data_byte=psum_dbyte
                data_type=psum_dtype
                ofm_len = ofm_shape.h * ofm_shape.w * ofm_shape.c * psum_dbyte
                self.psum_result_queue[icg] = self.unisram_onchip[self.psum_sram_base:self.psum_sram_base+ofm_len].copy().reshape(
                    ofm_shape.h, ofm_shape.c//self.ch_per_group, ofm_shape.w, self.ch_per_group*data_byte
                ).view(assign_np_dtype(data_type))

            if accu_en:
                assert self.psum_sram_base == psum_sram_base, "Psum sram base address is incorrect!!!"
                data_byte=psum_dbyte
                data_type=psum_dtype
                ofm_len = ofm_shape.h * ofm_shape.w * ofm_shape.c * psum_dbyte
                self.psum_fetch_queue[icg] = self.unisram_onchip[self.psum_sram_base:self.psum_sram_base+ofm_len].copy().reshape(
                    ofm_shape.h, ofm_shape.c//self.ch_per_group, ofm_shape.w, self.ch_per_group*data_byte
                ).view(assign_np_dtype(data_type))

                # conv_result_flatten is a np byte array
                conv_result_flatten = EmulateValidator.flatten_fm(conv_result, CIM_COL, psum_dtype)
                accu_len=conv_result_flatten.shape[0]
                assert accu_len == ofm_shape.h * ofm_shape.w * CIM_OC * psum_dbyte, "Shape of conv result isn't correct!!!"

                conv_result_unflatten = conv_result_flatten.reshape(ofm_shape.h, -1)
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
                # Reshape psum value into 3D array for easier debug usage
                data_byte=psum_dbyte
                data_type=psum_dtype
                ofm_len = ofm_shape.h * ofm_shape.w * ofm_shape.c * psum_dbyte
                self.psum_result_queue[icg] = self.unisram_onchip[self.psum_sram_base:self.psum_sram_base+ofm_len].reshape(
                    ofm_shape.h, ofm_shape.c//self.ch_per_group, ofm_shape.w, self.ch_per_group*data_byte
                ).copy().view(assign_np_dtype(data_type))

            if req_en and icg == cim_sic-1:
                # Check with the convolution result first
                self.ofm_sram_base = ofm_sram_base
                ofm_h, ofm_w = ofm_shape.h, ofm_shape.w
                ofm_result=self.callnode.ofm_data[
                    :,
                    oc_group*CIM_OC:oc_group*CIM_OC+CIM_OC,
                    o_tile_coord.h:o_tile_coord.h+ofm_h,
                    o_tile_coord.w:o_tile_coord.w+ofm_w
                ]
                ofm_result_flatten = EmulateValidator.flatten_fm(ofm_result, CIM_COL, psum_dtype)
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

                if not overwrite and not align_mode:
                    self.unisram_onchip[ofm_sram_base:ofm_sram_base+req_len] = req_result_ofm_dtype.copy()
                    ofm_h, ofm_w = ofm_shape.h, ofm_shape.w
                    ofm_result=self.callnode.req_data[
                        :,
                        oc_group*CIM_OC:oc_group*CIM_OC+CIM_OC,
                        o_tile_coord.h:o_tile_coord.h+ofm_h,
                        o_tile_coord.w:o_tile_coord.w+ofm_w
                    ]
                    ofm_result_flatten = EmulateValidator.flatten_fm(ofm_result, CIM_COL, ofm_dtype)
                    unisram_onchip = self.unisram_onchip[ofm_sram_base:ofm_sram_base+req_len]
                    assert (ofm_result_flatten==unisram_onchip).all(), "Requantization result stored in UNI_SRAM is incorrect!!!"

                    data_byte=ofm_dbyte
                    data_type=ofm_dtype
                    ofm_len = ofm_shape.h * ofm_shape.w * ofm_shape.c * data_byte
                    self.ofm = self.unisram_onchip[ofm_sram_base:ofm_sram_base+ofm_len].reshape(
                        ofm_shape.h, ofm_shape.c//self.ch_per_group, ofm_shape.w, self.ch_per_group*data_byte
                    ).copy().view(assign_np_dtype(data_type))

                elif overwrite and not align_mode:
                    for idx_h in range(ofm_shape.h):
                        for ocg in range(ofm_shape.c // CIM_COL):
                            for idx_w in range(ofm_shape.w):
                                dst = (
                                    idx_h * ofm_shape.c * ofm_shape.w +
                                    ocg * ofm_shape.w * CIM_COL +
                                    idx_w * CIM_COL
                                )* psum_dbyte + ofm_sram_base
                                src = (
                                    idx_h * ofm_shape.c * ofm_shape.w +
                                    ocg * ofm_shape.w * CIM_COL +
                                    idx_w * CIM_COL
                                )* ofm_dbyte
                                self.unisram_onchip[dst:dst+CIM_COL*ofm_dbyte] = req_result_ofm_dtype[src:src+CIM_COL*ofm_dbyte].copy()
                    data_byte=ofm_dbyte
                    data_type=ofm_dtype
                    self.ofm = req_result_ofm_dtype.reshape(
                        ofm_shape.h, ofm_shape.c//self.ch_per_group, ofm_shape.w, self.ch_per_group*data_byte
                    ).copy().view(assign_np_dtype(data_type))
                else:
                    assert False, f"Invalid overwrite or align_mode, overwrite={overwrite}, align_mode={align_mode}"
                
                
                unisram_onchip_np = self.unisram_onchip
                dump_dir = os.path.join(self.dump_dir, 'unisram_onchip')
                if not os.path.exists(dump_dir):
                    os.makedirs(dump_dir)
                npy_path = os.path.join(dump_dir, f"req{self.req_idx}.npy")
                np.save(npy_path,unisram_onchip_np)

        
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

        dump_dir = os.path.join(self.dump_dir, 'ofm_extern')
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        npy_path = os.path.join(dump_dir, f"req{self.req_idx}.npy")
        ofm_extern_np = self.ofm_extern
        np.save(npy_path,ofm_extern_np)
        self.req_idx+=1

    def compare_input_buffer(self, input_buffer: np.array, o_coord: Shape, icg: int, window_info: WindowInfo, k_size: int, ifm_c: int, ifm_shape: HWC, ofm_shape: HWC,):
        """
        Description
        -----------
        Compare the fetched input buffer with the extracted ifm golden data

        Parameters
        ----------
        input_buffer(8x64b), o_coord, icg: ic segment, window_info, k_size, ifm_c, ifm_shape(debug), ofm_shape(shape),
        """

        oh,ow = o_coord.h, o_coord.w
        kh, kw = window_info.kernel_shape.h, window_info.kernel_shape.w
        sh, sw = window_info.strides.h, window_info.strides.w

        h_start, w_start = oh*sh, ow*sw
        h_end, w_end = h_start+kh, w_start+kw
        ic_pivot = icg*self.ch_per_group if k_size == 8 else icg*self.ch_per_group*8
        ic_span = self.ch_per_group if k_size == 8 else self.ch_per_group*8
        ifm_extract = self.ifm[
            :,
            ic_pivot:ic_pivot+ic_span,
            h_start:h_end,
            w_start:w_end
        ].squeeze(0).numpy()
        ifm_extract = ifm_extract.reshape(ifm_extract.shape[0], -1)
        # ifm_extract is now of shape (c,k_size), where c is ic element within a segment
        ifm_extract = ifm_extract[:,:-1] if k_size == 8 else ifm_extract[:,-1:] 
        
        if k_size == 8:
            # ifm_extract is now of shape (k_size,c)
            ifm_extract = np.transpose(ifm_extract, (1,0))
        else:
            if ifm_extract.shape[0] < ic_span:
                pad_len = ic_span - ifm_extract.shape[0]
                pad_array = np.zeros((pad_len,k_size),dtype=ifm_extract.dtype)
                ifm_extract = np.concatenate([ifm_extract, pad_array], axis=0)
            ifm_extract = ifm_extract.reshape(ic_span)
            ifm_extract = ifm_extract.reshape(len(input_buffer),-1)

        packed_array = np.array(
            [pack_int8_to_int64(row) for row in ifm_extract], dtype=np.int64
        ).reshape(8) # packed_array is (8) int64 array
        if not (input_buffer == packed_array).all():
            print(f"ofm_shape={ofm_shape}, ifm_shape={ifm_shape}, oh={oh}, ow={ow}, ic_pivot={ic_pivot}, ic_span={ic_span}, k_size={k_size}, h_start={h_start}, h_end={h_end}, w_start={w_start}, w_end={w_end}")
            print(f"ifm_extract={packed_array}")
            print(f"input_buffer={input_buffer}")
        return (input_buffer == packed_array).all()
    
    def compare_psum_buffer(self, psum_buffer: np.array, o_coord: Shape, cim_oc: int, sp_group: int, icg: int, mode: bool):
        """
        Description
        -----------
        Compare the fetched psum buffer with the extracted psum data in vrf/store psum result inside

        Parameters
        ----------
        psum_buffer(cim_oc_x32b), o_coord, cim_oc, sp_group, mode
        mode == True, check for psum_fetch, else check for psum_result
        """
        psum_fetch_queue = self.psum_fetch_queue
        psum_result_queue = self.psum_result_queue
        # result_last_key = list(self.psum_result_queue.keys())[-1]
        # if self.psum_fetch_queue and len(self.psum_fetch_queue) > 0:
        #     fetch_last_key = list(self.psum_fetch_queue.keys())[-1]
        # else:
        #     fetch_last_key = None  # or handle default

        # if fetch_last_key is not None: assert icg >= 0 or icg <= fetch_last_key, f"out of bound access!!! icg={icg}, fetch_last_key={fetch_last_key}"
        # assert icg >= 0 or icg <= result_last_key, f"out of bound access!!! icg={icg}, result_last_key={result_last_key}"
        
        psum_gold = self.psum_fetch_queue[icg][
            o_coord.h,
            (sp_group*cim_oc)//self.ch_per_group:(sp_group*cim_oc+cim_oc)//self.ch_per_group,
            o_coord.w,
            :
        ] if mode else \
        self.psum_result_queue[icg][
            o_coord.h,
            (sp_group*cim_oc)//self.ch_per_group:(sp_group*cim_oc+cim_oc)//self.ch_per_group,
            o_coord.w,
            :
        ] 
        psum_gold = psum_gold.reshape(cim_oc)
        psum_buffer_extarct = psum_buffer[sp_group*cim_oc:sp_group*cim_oc+cim_oc]

        return (psum_buffer_extarct == psum_gold).all()
    
    def compare_bias_buffer(self, bias_buffer: np.array, cim_oc: int, sp_group: int):
        """
        Description
        -----------
        Compare the fetched bias buffer with the extracted bias data in vrf

        Parameters
        ----------
        bias_buffer(1x32b)
        """

        return (bias_buffer[sp_group*cim_oc:sp_group*cim_oc+cim_oc] == self.bias).all()

    def compare_weight(self, macro_weight: np.array, ping: bool, k_size: int):
        """
        Parameters
        ----------
            macro_weight is of shape (2,cim_segment,cim_row,cim_col) int8 np array
            
        """
        ping_idx = 0 if ping else 1
        cim_segment_= macro_weight.shape[1] # hardware constraint
        cim_row = macro_weight.shape[2]
        cim_col = macro_weight.shape[3]
        O, C, K = self.weight.shape
        weight_gold=self.weight
        ic_element=self.ch_per_group if k_size == 8 else cim_row
        # pad C dimension if C is less than 64 for 1x1 config
        if k_size == 1 and C < cim_row:
            pad_width = cim_row - C
            pad_tensor = np.zeros((O, pad_width, K), dtype=weight_gold.dtype)
            weight_gold = np.concatenate([weight_gold, pad_tensor], axis=1)
            C=cim_row
        # (O, C, K) → (oc_seg, cim_col, ic_seg, ic_element, K)
        weight_gold=weight_gold.reshape(O//cim_col,cim_col,C//ic_element,ic_element,K)
        # reorder to (oc_seg, ic_seg, K, ic_element, cim_col)
        weight_gold=torch.from_numpy(weight_gold).permute(0, 2, 4, 3, 1).numpy()
        # reorder to (oc_seg *ic_seg, ic_element*K, cim_col)
        cim_segment=weight_gold.shape[0]*weight_gold.shape[1]
        weight_gold = weight_gold.reshape(
            weight_gold.shape[0]*weight_gold.shape[1],
            weight_gold.shape[2]*weight_gold.shape[3],
            weight_gold.shape[4]
        )
        
        weight_extract = macro_weight[ping_idx,0:cim_segment,:,:]
        return (weight_extract == weight_gold).all()

    def compare_compute_result(self, mac_result: np.array, o_coord: Shape, icg: int):
        """
        Parameters
        ----------
            mac_result: shape of (oc_segment, cim_col) int32 np array
        """

        mac_result=mac_result.reshape(-1,) # flatten to 1D array
        mac_result_gold = self.mac_result_queue[icg][:,o_coord.h,o_coord.w]
        return (mac_result == mac_result_gold).all()
    
    def compare_store_result(
            self, 
            vrf_mem: np.array, 
            o_coord: Shape, 
            cim_oc: int, 
            sp_group: int, 
            ofm_sram_base: int, 
            psum_sram_base: int, 
            ofm_shape: Shape, 
            cim_col: int,
            psum_dtype,
            ofm_dtype,
            req_en: bool,
            overwrite: bool,
            align_mode: bool,
    ):
        ofm_gold = self.ofm[
            o_coord.h,
            (sp_group*cim_oc)//self.ch_per_group:(sp_group*cim_oc+cim_oc)//self.ch_per_group,
            o_coord.w,
            :
        ] if req_en else self.psum_result_queue[list(self.psum_result_queue.keys())[-1]][
            o_coord.h,
            (sp_group*cim_oc)//self.ch_per_group:(sp_group*cim_oc+cim_oc)//self.ch_per_group,
            o_coord.w,
            :
        ] 
        ofm_gold = ofm_gold.reshape(cim_oc)
        if not req_en:
            extract_data_bytes=stride_data_bytes = dtype_to_bytes(psum_dtype)
            dtype = psum_dtype
        else:
            stride_data_bytes = dtype_to_bytes(psum_dtype) if overwrite else dtype_to_bytes(ofm_dtype)
            extract_data_bytes = dtype_to_bytes(ofm_dtype)
            dtype = ofm_dtype
        data_arr=[]
        sram_base = ofm_sram_base if req_en else psum_sram_base
        for ocg in range(cim_oc//cim_col):# oc_segments
            addr = (
                o_coord.h * ofm_shape.c * ofm_shape.w + 
                sp_group * cim_oc * ofm_shape.w +
                ocg * ofm_shape.w * cim_col +
                o_coord.w * cim_col
            )* stride_data_bytes + sram_base
            data=vrf_mem[addr:addr+cim_col*extract_data_bytes].view(assign_np_dtype(dtype))
            data_arr.append(data)
        ofm_extact = np.array(data_arr).reshape(cim_oc)
        equal = ofm_extact == ofm_gold
        return (ofm_extact == ofm_gold).all()

    def tile_result_check(self, tile_attrs,ofm_extern):
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
                    ofm_extract[b8_idx:b8_idx+self.ch_per_group*ofm_dbyte] = torch.from_numpy(ofm_extern[addr:addr+self.ch_per_group*ofm_dbyte].copy())
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
        ).numpy()

        # ofm_gold is CHW ofm_dbyte
        ofm_gold = self.ofm_gold.squeeze(0)
        ofm_gold = ofm_gold[
            oc_group*o_tile_c:oc_group*o_tile_c+o_tile_c,
            o_tile_coord.h:o_tile_coord.h+o_tile_h,
            o_tile_coord.w:o_tile_coord.w+o_tile_w,
        ].numpy()
        equal = (ofm_gold==ofm_extract)
        assert (ofm_gold==ofm_extract).all(), "Store result isn't correct!!!"