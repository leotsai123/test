from .isa_def import CommonConfig, TLBR, HW


class WindowInfo(CommonConfig):
    def __init__(
        self,
        padding=TLBR(0, 0, 0, 0),
        upsample_ratio=HW(1, 1),
        kernel_shape=HW(1, 1),
        strides=HW(1, 1),
        dilation=HW(1, 1),
    ):
        self.padding = padding
        self.upsample_ratio = upsample_ratio
        self.kernel_shape = kernel_shape
        self.strides = strides
        self.dilation = dilation

    def emit_common_config(self, emitter):
        pass
