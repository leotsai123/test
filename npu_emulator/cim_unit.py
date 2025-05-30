import numpy as np 
from npu_emulator.utils import assign_np_dtype


class CimMacro:
    def __init__(self, cim_col: int = 8, cim_row: int = 64, cim_segment: int = 8,):
        self.cim_segment = cim_segment
        self.cim_col = cim_col
        self.cim_row = cim_row
        self.weight = np.zeros((2,cim_segment, cim_row, cim_col), dtype=np.int8) # 4KB*2 ping-pong macro

class CimUnit:
    def __init__(
        self,
        cim_sic: int=None, # configurable cim_oc, change its value based on macro op
        cim_soc: int=None, # configurable cim_ic, change its value based on macro op
        slide_size: int = 5, # Hardware constraint
        cim_oc_: int = 32 # Hardware constraint
    ):
        self.cim_soc = cim_soc 
        self.cim_sic = cim_sic 
        self.cim_oc_ = cim_oc_
        self.slide_size = slide_size
        self.cim_macro = CimMacro()
        self.psum_buffer = np.zeros((slide_size, cim_oc_), dtype=np.int32)

    def set_config(self, cim_soc: int, cim_sic: int):
        self.cim_soc = cim_soc
        self.cim_sic = cim_sic

    def compute(self, input_buffer: np.array, slide_idx: int, icg: int, ping: bool, sp_group: int, cim_oc: int, ifm_dtype):
        """
        Description
        -----------
            based on the given icg, iterate through the oc segments in cim_macro, and accumulate psum result in psum_buffer
        Parameters
        ----------
            input_buffer: 8 64bit registers
            slide_idx
            icg: ic segment
            ping
            sp_group
            cim_oc
            ifm_dtype
        """
        ping_idx = 0 if ping else 1
        result_arr=[]
        buff_base=sp_group*cim_oc
        for ocg in range(self.cim_soc): # oc segments
            seg_idx=ocg*self.cim_sic+icg
            weight=self.cim_macro.weight[ping_idx,seg_idx,:,:] # shape of (cim_row,cim_col)
            input=input_buffer.view(assign_np_dtype(ifm_dtype)).reshape(-1,self.cim_macro.cim_row) # split input buffer into 8x8 8bit registers
            # cim_macro compute, input is of shape(1,64), weight is of shape(64,8)
            result = input.astype(np.int32) @ weight.astype(np.int32)
            result_arr.append(result)
            # psum_buffer accumulation
            for col in range(self.cim_macro.cim_col):
                self.psum_buffer[slide_idx][buff_base+ocg*self.cim_macro.cim_col+col] += result[:,col]
        return np.array(result_arr)
