import numpy as np
from npu_emulator.cim_unit import CimMacro
from macro_gen.isa_def import MemType
from typing import Tuple

class VRF:
    def __init__(self, size: int = 16 * 1024):  # 16KB
        # view vrf as a 1D array with byte address
        self.mem = np.zeros(size, dtype=np.int8)

    def read(self, addr: int, length: int) -> np.array:
        return self.mem[addr:addr+length].clone()

    def write(self, addr: int, data: np.array):
        self.mem[addr:addr+len(data)] = data


class VPU:
    def __init__(self, vrf: VRF, cim_macro: CimMacro):
        self.vrf = vrf
        self.cim_macro = cim_macro

    def load_vrf(self, seg_num: int, v_stride: int, seg_stride: int, seg_len: int, src_base_addr: int, dst_base_addr: int, src_type: np.array):
        for i in range(seg_num):
            self.vrf.write(
                dst_base_addr+seg_len*i+v_stride*i,
                src_type[src_base_addr+i*seg_stride:src_base_addr+i*seg_stride+seg_len],
            )

    def load_weight(
        self, 
        soc: int, 
        oc_stride: int, 
        sic: int, 
        ic_stride: int, 
        eic: int, 
        k_stride: int, 
        src_base_addr: int, 
        src_len: int, 
        k_size: int, 
        src_type: np.array, 
        ping: bool,
        kernel_hw: Tuple
    ):
        
        k_h, k_w = kernel_hw[0], kernel_hw[1]
        seg_dst=0
        seg_idx=0
        for ocg in range(soc): # oc segments
            src_org = src_base_addr + ocg* oc_stride
            for icg in range(sic): # ic segments
                src = src_org + icg*ic_stride
                if k_size==1:
                    eic_step = (self.cim_macro.cim_row // k_size) if k_h == 1 and k_w == 1 else 1
                else:
                    eic_step = 1
                for eicg in range(0,eic,eic_step): # inner ic
                    
                    if k_size !=1:
                        seg_dst = eicg
                        last= eicg+1 == (self.cim_macro.cim_row//k_size)
                    else:
                        if k_h == 1 and k_w == 1: last = eicg+(self.cim_macro.cim_row // k_size) == eic
                        else: last = eicg+1 == eic
                    
                    # ----- Initiate weight programming ------ #
                    elements_per_column = src_len // self.cim_macro.cim_col
                    data = src_type[src:src+src_len]
                    data_byte=1
                    data = data.reshape(-1, self.cim_macro.cim_col*data_byte)
                    ping_idx = 0 if ping else 1
                    if ping_idx > 1 or seg_idx >= self.cim_macro.cim_segment or (seg_dst+elements_per_column) > self.cim_macro.cim_row:
                        raise ValueError(
                            f"invalid access of cim macro: ping_idx={ping_idx}, seg_idx={seg_idx}, seg_dst={seg_dst}, elements_per_column={elements_per_column}"
                        )
                    weight=self.cim_macro.weight[ping_idx,:,:,:]
                    if k_size==8:
                        for k in range(k_size):
                            self.cim_macro.weight[ping_idx,seg_idx,k*elements_per_column+seg_dst,:] = data[k]
                    else:
                        if last and seg_dst+elements_per_column < self.cim_macro.cim_row: # fill the rest of the cim_row with zeros
                            self.cim_macro.weight[ping_idx,seg_idx,seg_dst:seg_dst+elements_per_column,:] = data
                            self.cim_macro.weight[ping_idx,seg_idx,seg_dst+1:,:]=0
                        else:
                            self.cim_macro.weight[ping_idx,seg_idx,seg_dst:seg_dst+elements_per_column,:] = data
                    # ----- End weight programming ------ #
                    src+=k_stride
                    if k_size==1:
                        seg_dst += elements_per_column
                seg_idx+=1
                seg_dst=0

    def store_ofm(self, src, len, dst, dst_type):
        dst_type[dst:dst+len] = self.vrf.mem[src:src+len]
