from macro_gen.isa_def import ConvMacroOp, TLBR, PositionMap, MemType, Dtype
from model_construct.node import HWC, HW
from tiling.tiling import Shape
import math
import numpy as np

from npu_emulator.fetch_uinit import FetchUnit
from npu_emulator.cim_unit import CimUnit
from npu_emulator.store_unit import StoreUnit
from npu_emulator.validator import EmulateValidator



class TopController:
    def __init__(self, fetch_unit: FetchUnit, cim_unit: CimUnit, store_unit: StoreUnit, ch_per_group: int,): # params include all CONV macro_op fields
        self.fetch_unit = fetch_unit
        self.cim_unit = cim_unit
        self.store_unit = store_unit
        self.ch_per_group = ch_per_group
    
    def execute(self, params: ConvMacroOp, validator: EmulateValidator):
        ofm_h, ofm_w, ofm_c = params.ofm_shape.h, params.ofm_shape.w, params.ofm_shape.c
        ifm_h, ifm_w, ifm_c = params.ifm_shape.h, params.ifm_shape.w, params.ifm_shape.c
        # p_ifm_c = math.ceil(ifm_c/self.ch_per_group) * self.ch_per_group # p_ifm_c is already padded in macro_gen
        ifm_dtype = params.ifm_dtype
        weight_dtype = params.kernel_dtype
        bias_dtype=psum_dtype = params.psum_dtype
        cim_sic, cim_soc = params.cim_sic, params.cim_soc
        cim_ic, cim_oc = params.cim_ic, params.cim_oc # cim_oc = cim_soc * self.ch_per_group
        window_info = params.window_info
        kernel_shape = window_info.kernel_shape
        strides = window_info.strides
        padding = window_info.padding
        posmap = params.posmap
        ifm_sram_base = params.ifm_sram_base
        ofm_sram_base = params.ofm_sram_base
        psum_sram_base = params.psum_sram_base
        bias_sram_base = params.bias_sram_base
        bias_en = params.bias_en
        accu_en = params.accu_en
        req_en = params.req_en
        k_size = params.k_size
        sp_group = params.sp_group
        ping=params.ping
        scale_mode = params.scale_mode
        scale_mantissa = params.scale_mantissa
        scale_shift = params.scale_shift
        act_type = params.act_type
        act_max = params.act_max
        act_min = params.act_min
        ofm_dtype = params.ofm_dtype
        overwrite = params.overwrite
        align_mode = params.align_mode

        # ofm tile shape sanity check
        ofm_h_ = (ifm_h + padding.t + padding.b - (kernel_shape.h - 1) - 1) // strides.h + 1
        assert ofm_h == ofm_h_, "calculated ofm tile height is out of bound"
        ofm_w_ = (ifm_w + padding.l + padding.r - (kernel_shape.w - 1) - 1) // strides.w + 1
        assert ofm_w == ofm_w_, "calculated ofm tile width is out of bound"

        # operand precision sanity check
        assert ifm_dtype == Dtype.INT8 or ifm_dtype == Dtype.UINT8, "input precision isn't supported currently!!!"
        assert weight_dtype == Dtype.INT8, "weight precision isn't supported currently!!!"
        assert psum_dtype == Dtype.INT32, "psum precision isn't supported currently!!!"
        
        
        
        self.cim_unit.set_config(cim_soc=cim_soc, cim_sic=cim_sic)
        slide_size = self.cim_unit.slide_size
        bias_fetched=False
        
        assert validator.compare_weight(
            self.cim_unit.cim_macro.weight,
            ping,
            k_size
        ), "weight in cim_macro is incorrect!!!"

        #print(f"k_size={k_size}, cim_ic={cim_ic}, cim_oc={cim_oc}, accu_en={accu_en}, bias_en={bias_en}")
        for oh in range(ofm_h):
            for ow_ in range(0,ofm_w, slide_size):
                for icg in range(cim_sic): # ic segments
                    for s in range(slide_size):
                        ow = ow_ + s
                        if ow >= ofm_w: break
                        #print(f"oh={oh}, ow={ow}, icg={icg}, s={s}")
                        flush_en = s == 0 or ow == 0
                        fetch_input_request = {
                            "o_coord": Shape(oh,ow),
                            "ifm_shape": HWC(ifm_h, ifm_w, ifm_c), # NOTE: ifm_c is padded to align B8
                            "kernel_shape": kernel_shape,
                            "strides": strides,
                            "padding": padding,
                            "posmap": posmap,
                            "icg": icg,
                            "k_size": k_size,
                            "ifm_sram_base":  ifm_sram_base,
                            "ifm_dtype": ifm_dtype,
                            "flush_en": flush_en
                        }
                        input_buffer = self.fetch_unit.fetch_input(fetch_input_request)
                        assert validator.compare_input_buffer(
                            input_buffer,
                            Shape(oh,ow),
                            icg,
                            window_info,
                            k_size,
                            ifm_c,
                            HWC(ifm_h, ifm_w, ifm_c), # debug
                            HWC(ofm_h, ofm_w, ofm_c), # debug
                        ), "input buffer content incorrect!!!"

                        # one conv_op needs to fetch bias only once
                        if bias_en and not bias_fetched:
                            fetch_bias_request = {
                                "bias_sram_base": bias_sram_base,
                                "cim_oc": cim_oc,
                                "sp_group": sp_group,
                                "bias_dtype": bias_dtype
                            }
                            bias_buffer=self.fetch_unit.fetch_bias(fetch_bias_request, validator.bias)
                            bias_fetched = True
                            assert validator.compare_bias_buffer(
                                bias_buffer,
                                cim_oc,
                                sp_group,
                            ), "bias buffer content incorrect!!!"
                        if accu_en and icg==0:
                            fetch_psum_request = {
                                "o_coord": Shape(oh,ow),
                                "ofm_shape": HWC(ofm_h, ofm_w, ofm_c),
                                "psum_sram_base": psum_sram_base,
                                "bias_sram_base": bias_sram_base,
                                "cim_oc": cim_oc,
                                "sp_group": sp_group,
                                "psum_dtype": psum_dtype,
                            }
                            self.cim_unit.psum_buffer[s] = self.fetch_unit.fetch_psum(request=fetch_psum_request, cim_oc_=self.cim_unit.cim_oc_,)
                            
                            assert validator.compare_psum_buffer(
                                psum_buffer=self.cim_unit.psum_buffer[s],
                                o_coord=Shape(oh,ow),
                                cim_oc=cim_oc,
                                sp_group=sp_group,
                                icg=icg,
                                mode=True,
                            ), "fetched psum buffer content incorrect!!!"

                        if bias_en and icg == 0:
                            self.cim_unit.psum_buffer[s][sp_group*cim_oc:sp_group*cim_oc+cim_oc] = bias_buffer[sp_group*cim_oc:sp_group*cim_oc+cim_oc]
                        mac_result=self.cim_unit.compute(
                            input_buffer,
                            s,
                            icg,
                            ping,
                            sp_group,
                            cim_oc,
                            ifm_dtype,
                        )
                        assert validator.compare_compute_result(
                            mac_result,
                            Shape(oh,ow),
                            icg
                        ), "mac_result incorrect!!!"
                        
                        assert validator.compare_psum_buffer(
                            psum_buffer=self.cim_unit.psum_buffer[s],
                            o_coord=Shape(oh,ow),
                            cim_oc=cim_oc,
                            sp_group=sp_group,
                            icg=icg,
                            mode=False,
                        ), "accumulated psum buffer content incorrect!!!"

                        if icg == cim_sic-1:
                            psum_buffer=self.cim_unit.psum_buffer[s]
                            store_request={
                                "psum_buffer":      psum_buffer,
                                "o_coord":          Shape(oh,ow),
                                "ofm_shape":        HWC(ofm_h, ofm_w, ofm_c),
                                "cim_oc":           cim_oc,
                                "sp_group":         sp_group,
                                "req_en":           req_en,
                                "scale_mode":       scale_mode,
                                "scale_mantissa":   scale_mantissa,
                                "scale_shift":      scale_shift,
                                "act_type":         act_type,
                                "act_max":          act_max,
                                "act_min":          act_min,
                                "ofm_dtype":        ofm_dtype,
                                "psum_dtype":       psum_dtype,
                                "ofm_sram_base":    ofm_sram_base,
                                "psum_sram_base":   psum_sram_base,
                                "overwrite":        overwrite,
                                "align_mode":       align_mode,
                            }
                            
                            self.store_unit.store_result(store_request)

                            assert validator.compare_store_result(
                                vrf_mem=self.store_unit.vrf.mem,
                                o_coord=Shape(oh,ow),
                                cim_oc=cim_oc,
                                sp_group=sp_group,
                                ofm_sram_base=ofm_sram_base,
                                psum_sram_base=psum_sram_base,
                                ofm_shape=HWC(ofm_h, ofm_w, ofm_c),
                                cim_col=self.cim_unit.cim_macro.cim_col,
                                psum_dtype=psum_dtype,
                                ofm_dtype=ofm_dtype,
                                req_en=req_en,
                                overwrite=overwrite,
                                align_mode=align_mode,
                            ), "store_result incorrect!!!"
