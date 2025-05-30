# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 14:10:38 2025

@author: user
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from .layout import to_nchwb8, to_ohwcb8
from model_construct.node import Node, OCHW, HWC, XY
import torch
from model_construct.op_attrs import (
    CONV2D_OP
)
from tiling.tiling import TileAttrsBuilder
from macro_gen.macro_gen import MacroOpGen
from npu_emulator.validator import EmulateValidator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--emulator', type=str, default='low', help='Choose emulator: high or low')
args = parser.parse_args()
from npu_emulator.emulator import EmulateBuilder
from macro_gen.isa_def import LoadCimMacroOp, LoadMacroOp_, ConvMacroOp, LoadMicroOp_
from typing import Union

build_dir = "../build/"
if not os.path.exists(build_dir):
    os.makedirs(build_dir)


ifm_shapes = [
    HWC(h=int(line[2]), w=int(line[3]), c=int(line[1]))
    # for line in [l.split() for l in open("../model_data/TinyYOLOv7_pose/model_file/IFM.txt").readlines()[1:]]
    for line in [l.split() for l in open("../model_data/TinyYOLOv7_pose/model_file/IFM_div_20.txt").readlines()[1:]]
]

kernel_shapes = [
    {
        "k": int(line[0]),
        "ic": int(line[1]),
        "oc": int(line[2]),
        "stride": int(line[3])
    }
    for line in [l.split() for l in open("../model_data/TinyYOLOv7_pose/model_file/model.txt").readlines()[1:]]
]

# Sanity check
assert len(ifm_shapes) == len(kernel_shapes), "Mismatch between IFM and kernel layers."

# Construct layers
model_layers = []
outputs = []
for i, (ifm, weight) in enumerate(zip(ifm_shapes, kernel_shapes)):
    if i >= 0: # change i condition to select partial test
        kernel = OCHW(
            o=weight["oc"],
            c=weight["ic"],
            h=weight["k"],
            w=weight["k"]
        )

        ifm_dtype="int8"
        kernel_dtype="int8"
        bias_dtype="int32"
        psum_dtype="int32"
        ofm_dtype="uint8"

        act_min, act_max = Node.assign_value_range(ifm_dtype)
        kern_min, kern_max = Node.assign_value_range(kernel_dtype)

        ifm_data = torch.randint(act_min, act_max, (1, ifm.c, ifm.h, ifm.w), dtype=torch.float32)
        kernel_data = torch.randint(kern_min, kern_max, (kernel.o, kernel.c, kernel.h, kernel.w), dtype=torch.float32)
        # bias = torch.randint(-128, 128, (kernel.o,), dtype=torch.float32)
        bias = torch.arange(1, kernel.o + 1, dtype=torch.float32)
        scale_mantissa = torch.randint(0, 256, (1,), dtype=torch.float32)
        scale_shift = torch.randint(15, 17, (1,), dtype=torch.float32)

        stride = XY(weight["stride"], weight["stride"])
        padding = [1,1,1,1] if weight["k"] == 3 else [0,0,0,0]
        activation="ReLU"


        layer = Node(
            name=f"conv{i}",
            op=CONV2D_OP,
            ifm_shape=ifm,
            kernel_shape=kernel,
            ifm_data=ifm_data,
            kernel_data=kernel_data,
            stride=stride,
            padding=padding,
            bias=bias,
            scale_mantissa=scale_mantissa,
            scale_shift=scale_shift,
            ifm_dtype=ifm_dtype,
            ofm_dtype=ofm_dtype,
            kernel_dtype=kernel_dtype,
            bias_dtype=bias_dtype,
            psum_dtype=psum_dtype,
            activation=activation
        )
        # Perform convolutions
        outputs.append(layer.conv2d())
        model_layers.append(layer)

        tiler = TileAttrsBuilder(ch_per_group=8)
        node_tile_attrs_map = tiler(layer)
        ch_per_group=8
        gen_macro_ops = MacroOpGen(node_tile_attrs_map[layer], ch_per_group, mode=args.emulator)
        macro_ops = gen_macro_ops(layer)

        macro=True
        if not os.path.exists(f"{build_dir}/macro_ops"):
            os.makedirs(f"{build_dir}/macro_ops")
        with open(f"{build_dir}/macro_ops/macro_op{i}.log", 'w') as f:
            for idx, macro_op in enumerate(macro_ops):
                if args.emulator == "low" and macro:
                    if isinstance(macro_op, Union[LoadMacroOp_,LoadCimMacroOp, ConvMacroOp]):
                        f.write(f"macro_{idx}: {macro_op}\n")
                elif args.emulator == "low" and not macro:
                    if isinstance(macro_op, Union[LoadMicroOp_,LoadCimMacroOp, ConvMacroOp]):
                        f.write(f"macro_{idx}: {macro_op}\n")

        print(f"emulating layer {i}, ifm shape: (C,H,W):({ifm.c},{ifm.h},{ifm.w}), weight shape: (O,C,H,W): ({kernel.o},{kernel.c},{kernel.h},{kernel.w}), stride: {stride}, activation: {activation}")
        
        # initialize validator
        validator = EmulateValidator(
            macro_ops=macro_ops,
            callnode=layer,
            ch_per_group=ch_per_group, 
            layer_idx=i
        )
       
        # hardware constraints
        slide_size=5
        cim_oc_ = 32
        emulator = EmulateBuilder(
            callnode=layer,
            macro_ops=macro_ops,
            ch_per_group=ch_per_group,
            slide_size=slide_size,
            cim_oc_=cim_oc_,
            layer_idx=i,
            validator=validator,
            mode=args.emulator
        )
        emulator()





