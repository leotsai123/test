from typing import Any
from os import path, makedirs
from .op import MacroOp
from .isa_def import *
from tvm.runtime.contrib.novella.utils import get_artifacts_dir

MAJOR_VERSION = 0
MINOR_VERSION = 0
REVISION = 9


class Emitter:
    def __init__(self):
        self.cu_config_reg = CuConfigReg()
        self.ppu_config_reg = PpuConfigReg()
        self.load_config_reg = LoadConfigReg()
        self.store_config_reg = StoreConfigReg()
        self.target = None
        self.insts = []
        self.emitted_target = 0
        self.emitted_conv = 0
        self.emitted_dw_conv = 0
        self.emitted_pool = 0
        self.emitted_bin_eltwise = 0
        self.emitted_un_eltwise = 0
        self.emitted_reduce = 0
        self.emitted_load = 0
        self.emitted_store = 0
        self.emitted_barrier = 0
        self.emitted_ext_sync = 0
        self.emitted_config = 0
        self.macro_idx = 0

    def __call__(self, macro_ops):
        self.insts.append(REVISION << 24 | MINOR_VERSION << 16 | MAJOR_VERSION << 8 | 0xFF)
        for macro_op in macro_ops:
            macro_op.emit(self)
        return self.insts

    def dump_regs(self):
        arc_state_dir = f"{get_artifacts_dir()}arch_state/"
        if not path.exists(arc_state_dir):
            makedirs(arc_state_dir)
        with open(f"{arc_state_dir}macro_{self.macro_idx}.toml", "w") as f:
            f.write(f"[CuConfigReg]\n")
            f.write(f"{self.cu_config_reg}\n")
            f.write(f"[PpuConfigReg]\n")
            f.write(f"{self.ppu_config_reg}\n")
            f.write(f"[LoadConfigReg]\n")
            f.write(f"{self.load_config_reg}\n")
            f.write(f"[StoreConfigReg]\n")
            f.write(f"{self.store_config_reg}\n")
        self.macro_idx += 1

    def emit(self, op_code, conf):
        self.insts.append(int(op_code.value + conf * (2**8)))
        if op_code == OpCode.SET_TARGET:
            self.emitted_target += 1
        elif op_code == OpCode.CONV:
            self.emitted_conv += 1
            self.dump_regs()
        elif op_code == OpCode.DW_CONV:
            self.emitted_dw_conv += 1
            self.dump_regs()
        elif op_code == OpCode.POOL:
            self.emitted_pool += 1
            self.dump_regs()
        elif op_code == OpCode.BIN_ELTWISE:
            self.emitted_bin_eltwise += 1
            self.dump_regs()
        elif op_code == OpCode.UN_ELTWISE:
            self.emitted_un_eltwise += 1
            self.dump_regs()
        elif op_code == OpCode.LOAD:
            self.emitted_load += 1
            self.dump_regs()
        elif op_code == OpCode.STORE:
            self.emitted_store += 1
            self.dump_regs()
        elif op_code == OpCode.BARRIER:
            self.emitted_barrier += 1
        elif op_code == OpCode.EXT_SYNC:
            self.emitted_ext_sync += 1
        else:
            self.emitted_config += 1

    def set_target(self, target):
        if target != self.target:
            self.emit(OpCode.SET_TARGET, target.value)
            self.target = target

    def __get_depend_tag_val(self, dep_tag: DependencyTag):
        st_tag = 0 if dep_tag.st_tag == 0 else (dep_tag.st_tag - 1) % 15 + 1
        ppu_tag = 0 if dep_tag.ppu_tag == 0 else (dep_tag.ppu_tag - 1) % 15 + 1
        ctu_tag = 0 if dep_tag.ctu_tag == 0 else (dep_tag.ctu_tag - 1) % 15 + 1
        ld_tag = 0 if dep_tag.ld_tag == 0 else (dep_tag.ld_tag - 1) % 15 + 1

        return st_tag * (2**12) + ppu_tag * (2**8) + ctu_tag * (2**4) + ld_tag

    def emit_conv(self, accu_en, bias_en, req_en, dep_tag):
        conf = 0
        if accu_en:
            conf += 1
        if req_en:
            conf += 2
        if bias_en:
            conf += 4
        conf += self.__get_depend_tag_val(dep_tag) << 8

        self.emit(OpCode.CONV, conf)

    def emit_dw_conv(self, accu_en, bias_en, req_en, dep_tag):
        conf = 0
        if accu_en:
            conf += 1
        if req_en:
            conf += 2
        if bias_en:
            conf += 4
        conf += self.__get_depend_tag_val(dep_tag) << 8

        self.emit(OpCode.DW_CONV, conf)

    def emit_pool(self, pool_op_type, accu_en, req_en, dep_tag):
        conf = 0
        if accu_en:
            conf += 1
        if req_en:
            conf += 2
        if pool_op_type == PoolOpType.MAX_POOL:
            conf += 2**4
        conf += self.__get_depend_tag_val(dep_tag) << 8

        self.emit(OpCode.POOL, conf)

    def emit_bin_eltwise(self, bin_eltwise_op_type, broadcast_h, broadcast_w, broadcast_c, dep_tag):
        conf = bin_eltwise_op_type.value
        if broadcast_h:
            conf += 2**4
        if broadcast_w:
            conf += 2**5
        if broadcast_c:
            conf += 2**6
        conf += self.__get_depend_tag_val(dep_tag) << 8

        self.emit(OpCode.BIN_ELTWISE, conf)

    def emit_un_eltwise(self, un_eltwise_op_type, dep_tag):
        conf = un_eltwise_op_type.value
        conf += self.__get_depend_tag_val(dep_tag) << 8

        self.emit(OpCode.UN_ELTWISE, conf)

    def emit_load(self, dep_tag):
        conf = self.__get_depend_tag_val(dep_tag) << 8

        self.emit(OpCode.LOAD, conf)

    def emit_store(self, transpose, dep_tag):
        conf = 0
        if transpose:
            conf = 1
        conf += self.__get_depend_tag_val(dep_tag) << 8

        self.emit(OpCode.STORE, conf)

    def emit_barrier(self):
        self.emit(OpCode.BARRIER, 0)

    def emit_ext_sync(self):
        self.emit(OpCode.EXT_SYNC, 0)


class CuConfigReg:
    def __init__(self):
        self.ifm_config_reg = IfmConfigReg(True)
        self.weight_config_reg = WeightConfigReg()
        self.ofm_config_reg = OfmConfigReg()
        self.quantization_config_reg = QunatizationConfigReg()
        self.misc_config_reg = MiscConfigReg(True)

    def __str__(self):
        ret_val = ""
        ret_val += f"{self.ifm_config_reg}"
        ret_val += f"{self.weight_config_reg}"
        ret_val += f"{self.ofm_config_reg}"
        ret_val += f"{self.quantization_config_reg}"
        ret_val += f"{self.misc_config_reg}"
        return ret_val


class PpuConfigReg:
    def __init__(self):
        self.ifm_config_reg = IfmConfigReg()
        self.weight_config_reg = WeightConfigReg()
        self.ofm_config_reg = OfmConfigReg()
        self.quantization_config_reg = QunatizationConfigReg()
        self.misc_config_reg = MiscConfigReg()

    def __str__(self):
        ret_val = ""
        ret_val += f"{self.ifm_config_reg}"
        ret_val += f"{self.weight_config_reg}"
        ret_val += f"{self.ofm_config_reg}"
        ret_val += f"{self.quantization_config_reg}"
        ret_val += f"{self.misc_config_reg}"
        return ret_val


class LoadConfigReg:
    def __init__(self):
        self.dma_config_reg = DmaConfigReg()

    def __str__(self):
        ret_val = ""
        ret_val += f"{self.dma_config_reg}"
        return ret_val


class StoreConfigReg:
    def __init__(self):
        self.dma_config_reg = DmaConfigReg()

    def __str__(self):
        ret_val = ""
        ret_val += f"{self.dma_config_reg}"
        return ret_val


class RegList:
    def __init__(self):
        self.reg_list = {}

    def emit_common(self, emitter, op_name, conf):
        emitter.emit(op_name, conf)

    def get_region_val(self, op_name, region):
        return region[1][1] * (2**8) + region[1][0] * (2**4) + region[0].value

    def get_tlbr_val(self, tlbr):
        return tlbr.r * (2**12) + tlbr.l * (2**8) + tlbr.b * (2**4) + tlbr.t

    def get_hw_val(self, hw, is_upsampling=False):
        if is_upsampling:
            return hw.h * (2**4) + hw.w

        return hw.h * (2**8) + hw.w

    def get_dtype_val(self, dtypes):
        dtype_val = 0
        for i in range(5):
            dtype_val += dtypes[i].value * (2 ** (i * 4)) if dtypes[i] is not None else 0
        return dtype_val

    def get_quant_mode_val(self, quant_mode):
        reg_val = quant_mode[0].value
        for i in range(1, 4):
            if quant_mode[i]:
                reg_val += 2 ** ((i - 1) + 8)
        return reg_val

    def __str__(self):
        ret_val = ""
        max_len = 20
        for k, v in self.reg_list.items():
            if (
                k == OpCode.IFM_REGION
                or k == OpCode.IFM2_REGION
                or k == OpCode.OFM_REGION
                or k == OpCode.SRC_REGION
                or k == OpCode.DST_REGION
                or k == OpCode.WEIGHT_REGION
            ):
                val_str = f"0x{k.value:X} = {self.get_region_val(k, v)}"
            elif k == OpCode.IFM_PAD:
                val_str = f"0x{k.value:X} = {self.get_tlbr_val(v)}"
            elif (
                k == OpCode.IFM_UPSAMPLING_RATIO
                or k == OpCode.KERNEL_SHAPE
                or k == OpCode.KERNEL_STRIDE
                or k == OpCode.KERNEL_DILATION
            ):
                val_str = f"0x{k.value:X} = {self.get_hw_val(v, k == OpCode.IFM_UPSAMPLING_RATIO)}"
            elif k == OpCode.DATA_TYPE:
                val_str = f"0x{k.value:X} = {self.get_dtype_val(v)}"
            elif k == OpCode.QUANTIZATION_MODE:
                val_str = f"0x{k.value:X} = {self.get_quant_mode_val(v)}"
            elif k == OpCode.ACTIVATION_FUNCTION_TYPE:
                val_str = f"0x{k.value:X} = {v.value}"
            else:
                val_str = f"0x{k.value:X} = {v}"

            ret_val += f"{val_str:{max_len}} # {k} = {v}\n"

        return ret_val


class DmaConfigReg(RegList):
    def __init__(self):
        super().__init__()
        self.reg_list[OpCode.X_LENGTH] = 8
        self.reg_list[OpCode.Y_LENGTH] = 1
        self.reg_list[OpCode.Z_LENGTH] = 1
        self.reg_list[OpCode.SRC_BASE] = 0
        self.reg_list[OpCode.DST_BASE] = 0
        self.reg_list[OpCode.SRC_X_STRIDE] = 0
        self.reg_list[OpCode.DST_X_STRIDE] = 0
        self.reg_list[OpCode.SRC_Y_STRIDE] = 0
        self.reg_list[OpCode.DST_Y_STRIDE] = 0
        self.reg_list[OpCode.SRC_REGION] = [Region.DRAM, [0, 1]]
        self.reg_list[OpCode.DST_REGION] = [Region.DRAM, [0, 1]]

    def emit(self, emitter, config):
        for op_name, conf in config.items():
            if self.reg_list[op_name] == conf:
                continue
            if op_name == OpCode.SRC_REGION or op_name == OpCode.DST_REGION:
                if (
                    self.reg_list[op_name][0] == Region.DRAM
                    and self.reg_list[op_name][0] == conf[0]
                ):
                    continue
                reg_val = self.get_region_val(op_name, conf)
            else:
                reg_val = conf
            self.emit_common(emitter, op_name, reg_val)
            self.reg_list[op_name] = conf


class IfmConfigReg(RegList):
    def __init__(self, is_ctu=False):
        super().__init__()
        self.reg_list[OpCode.IFM_HEIGHT] = 1
        self.reg_list[OpCode.IFM_WIDTH] = 1
        self.reg_list[OpCode.IFM_CHANNEL] = 1
        self.reg_list[OpCode.IFM_PAD] = TLBR(0, 0, 0, 0)
        self.reg_list[OpCode.IFM_UPSAMPLING_RATIO] = HW(1, 1)
        self.reg_list[OpCode.IFM_ZERO_POINT] = 0
        self.reg_list[OpCode.IFM_BASE] = 0
        if not is_ctu:
            self.reg_list[OpCode.IFM2_BASE] = 0
        self.reg_list[OpCode.IFM_REGION] = [Region.I_SRAM, [0, 1]]
        if not is_ctu:
            self.reg_list[OpCode.IFM2_REGION] = [Region.I_SRAM, [0, 1]]

    def emit(self, emitter, config):
        for op_name, conf in config.items():
            if self.reg_list[op_name] == conf:
                continue
            if op_name == OpCode.IFM_REGION or op_name == OpCode.IFM2_REGION:
                if (
                    self.reg_list[op_name][0] == Region.DRAM
                    and self.reg_list[op_name][0] == conf[0]
                ):
                    continue
                reg_val = self.get_region_val(op_name, conf)
            elif op_name == OpCode.IFM_PAD:
                reg_val = self.get_tlbr_val(conf)
            elif op_name == OpCode.IFM_UPSAMPLING_RATIO:
                reg_val = self.get_hw_val(conf, True)
            else:
                reg_val = conf
            self.emit_common(emitter, op_name, reg_val)
            self.reg_list[op_name] = conf


class WeightConfigReg(RegList):
    def __init__(self):
        super().__init__()
        self.reg_list[OpCode.KERNEL_SHAPE] = HW(1, 1)
        self.reg_list[OpCode.KERNEL_STRIDE] = HW(1, 1)
        self.reg_list[OpCode.KERNEL_DILATION] = HW(1, 1)
        self.reg_list[OpCode.WEIGHT_ZERO_POINT] = 0
        self.reg_list[OpCode.TOTAL_IFM_CHANNEL] = 1
        self.reg_list[OpCode.WEIGHT_BASE] = 0
        self.reg_list[OpCode.WEIGHT_REGION] = [Region.DRAM, [0, 1]]

    def emit(self, emitter, config):
        for op_name, conf in config.items():
            if self.reg_list[op_name] == conf:
                continue
            if op_name == OpCode.WEIGHT_REGION:
                if (
                    self.reg_list[op_name][0] == Region.DRAM
                    and self.reg_list[op_name][0] == conf[0]
                ):
                    continue
                reg_val = self.get_region_val(op_name, conf)
            elif (
                op_name == OpCode.KERNEL_SHAPE
                or op_name == OpCode.KERNEL_STRIDE
                or op_name == OpCode.KERNEL_DILATION
            ):
                reg_val = self.get_hw_val(conf)
            else:
                reg_val = conf
            self.emit_common(emitter, op_name, reg_val)
            self.reg_list[op_name] = conf


class OfmConfigReg(RegList):
    def __init__(self):
        super().__init__()
        self.reg_list[OpCode.OFM_HEIGHT] = 1
        self.reg_list[OpCode.OFM_WIDTH] = 1
        self.reg_list[OpCode.OFM_CHANNEL] = 1
        self.reg_list[OpCode.OFM_ZERO_POINT] = 0
        self.reg_list[OpCode.OFM_BASE] = 0
        self.reg_list[OpCode.OFM_REGION] = [Region.O_SRAM, [0, 4]]

    def emit(self, emitter, config):
        for op_name, conf in config.items():
            if self.reg_list[op_name] == conf:
                continue
            if op_name == OpCode.OFM_REGION:
                if (
                    self.reg_list[op_name][0] == Region.DRAM
                    and self.reg_list[op_name][0] == conf[0]
                ):
                    continue
                reg_val = self.get_region_val(op_name, conf)
            else:
                reg_val = conf
            self.emit_common(emitter, op_name, reg_val)
            self.reg_list[op_name] = conf


class QunatizationConfigReg(RegList):
    def __init__(self):
        super().__init__()
        self.reg_list[OpCode.QUANTIZATION_MODE] = [
            ScaleModeType.PER_TENSOR_AFFINE,
            False,
            False,
            False,
        ]  # [mode, ifm_zp, weight_zp, ofm_zp]
        self.reg_list[OpCode.ACTIVATION_FUNCTION_TYPE] = ActivationType.CLIP
        self.reg_list[OpCode.CLIP_MIN] = 0
        self.reg_list[OpCode.CLIP_MAX] = 0
        self.reg_list[OpCode.SCALE_MANTISSA] = 0
        self.reg_list[OpCode.SCALE_SHIFT] = 0
        self.reg_list[OpCode.BIAS_BASE] = 0
        self.reg_list[OpCode.SCALE_BASE] = 0

    def emit(self, emitter, config):
        for op_name, conf in config.items():
            if self.reg_list[op_name] == conf:
                continue
            if op_name == OpCode.QUANTIZATION_MODE:
                if not any(i != j for i, j in zip(self.reg_list[OpCode.QUANTIZATION_MODE], conf)):
                    continue
                quant_mode_val = self.get_quant_mode_val(conf)
                self.emit_common(emitter, op_name, quant_mode_val)
            elif op_name == OpCode.ACTIVATION_FUNCTION_TYPE:
                self.emit_common(emitter, op_name, conf.value)
            elif op_name == OpCode.SCALE_BASE and conf == -1:
                continue
            elif op_name == OpCode.SCALE_MANTISSA and config[OpCode.SCALE_BASE] != -1:
                continue
            elif op_name == OpCode.SCALE_SHIFT and config[OpCode.SCALE_BASE] != -1:
                continue

            else:
                self.emit_common(emitter, op_name, conf)
            self.reg_list[op_name] = conf


class MiscConfigReg(RegList):
    def __init__(self, is_ctu=False):
        super().__init__()
        self.reg_list[OpCode.DATA_TYPE] = [
            Dtype.INT8,
            Dtype.INT8,
            Dtype.INT8,
            Dtype.INT32,
            Dtype.INT8,
        ]  # [ifm, weight, ofm, psum]
        if not is_ctu:
            self.reg_list[OpCode.IMM_LOW] = 0
            self.reg_list[OpCode.IMM_HIGH] = 0

    def emit(self, emitter, config):
        for op_name, conf in config.items():
            if self.reg_list[op_name] == conf or conf is None:
                continue
            if op_name == OpCode.DATA_TYPE:
                if not any(
                    i != j and i is not None for i, j in zip(self.reg_list[OpCode.DATA_TYPE], conf)
                ):
                    continue
                dtype_val = self.get_dtype_val(conf)
                emitter.emit(op_name, dtype_val)
            else:
                self.emit_common(emitter, op_name, conf)
            self.reg_list[op_name] = conf
