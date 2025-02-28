

import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.ir


# python3 nndeploy/op/op.py


class Op(_C.op.Op):
    def __init__(self):
        super().__init__()

    def set_name(self, name):
        return super().set_name(name)

    def get_name(self):
        return super().get_name()

    def set_op_type(self, op_type):
        return super().set_op_type(op_type)

    def get_op_type(self):
        return super().get_op_type()

    def set_param(self, param):
        return super().set_param(param)

    def get_param(self):
        return super().get_param()

    def set_device_type(self, device_type):
        return super().set_device_type(device_type)

    def get_device_type(self):
        return super().get_device_type()

    def set_stream(self, stream):
        return super().set_stream(stream)

    def get_stream(self):
        return super().get_stream()

    def set_precision_type(self, precision_type):
        return super().set_precision_type(precision_type)

    def get_precision_type(self):
        return super().get_precision_type()

    def get_input_name(self, index=0):
        return super().get_input_name(index)

    def get_output_name(self, index=0):
        return super().get_output_name(index)

    def get_input(self, index=0):
        return super().get_input(index)

    def get_output(self, index=0):
        return super().get_output(index)

    def set_input(self, tensor, index=0):
        return super().set_input(tensor, index)

    def set_output(self, tensor, index=0):
        return super().set_output(tensor, index)

    def set_all_input_name(self, names):
        return super().set_all_input_name(names)

    def set_all_output_name(self, names):
        return super().set_all_output_name(names)

    def get_all_input_name(self):
        return super().get_all_input_name()

    def get_all_output_name(self):
        return super().get_all_output_name()

    def get_all_input(self):
        return super().get_all_input()

    def get_all_output(self):
        return super().get_all_output()

    def rm_input(self, index):
        return super().rm_input(index)

    def set_all_input(self, tensors):
        return super().set_all_input(tensors)

    def set_all_output(self, tensors):
        return super().set_all_output(tensors)

    def get_constructed(self):
        return super().get_constructed()

    def set_parallel_type(self, parallel_type):
        return super().set_parallel_type(parallel_type)

    def get_parallel_type(self):
        return super().get_parallel_type()

    def set_inner_flag(self, flag):
        return super().set_inner_flag(flag)

    def set_initialized_flag(self, flag):
        return super().set_initialized_flag(flag)

    def get_initialized(self):
        return super().get_initialized()

    def set_time_profile_flag(self, flag):
        return super().set_time_profile_flag(flag)

    def get_time_profile_flag(self):
        return super().get_time_profile_flag()

    def set_debug_flag(self, flag):
        return super().set_debug_flag(flag)

    def get_debug_flag(self):
        return super().get_debug_flag()

    def set_running_flag(self, flag):
        return super().set_running_flag(flag)

    def is_running(self):
        return super().is_running()

    def infer_data_type(self):
        return super().infer_data_type()

    def infer_shape(self):
        return super().infer_shape()

    def infer_data_format(self):
        return super().infer_data_format()

    def init(self):
        return super().init()

    def deinit(self):
        return super().deinit()

    def reshape(self):
        return super().reshape()

    def pre_run(self):
        return super().pre_run()

    def get_workspace_size(self):
        return super().get_workspace_size()

    def set_workspace(self, workspace):
        return super().set_workspace(workspace)

    def get_flops(self):
        return super().get_flops()

    def check_or_alloc_output(self):
        return super().check_or_alloc_output()

    def run(self):
        raise NotImplementedError("Subclass must implement the run function")

    def post_run(self):
        return super().post_run()


class OpCreator(_C.op.OpCreator):
    def __init__(self):
        super().__init__()

    def create_op(self, device_type: nndeploy.base.DeviceType, name: str, op_type: nndeploy.ir.OpType, inputs: list[str], outputs: list[str]):
        raise NotImplementedError("Subclass must implement the create_op function")

    def create_op_shared_ptr(self, device_type: nndeploy.base.DeviceType, name: str, op_type: nndeploy.ir.OpType, inputs: list[str], outputs: list[str]):
        raise NotImplementedError("Subclass not need implement the create_op_shared_ptr function")


def register_op_creator(device_type_code: nndeploy.base.DeviceTypeCode, op_type: nndeploy.ir.OpType, creator: OpCreator):
    return _C.op.register_op_creator(device_type_code, op_type, creator)


def create_op(device_type: nndeploy.base.DeviceType, name: str, op_type: nndeploy.ir.OpType, inputs: list[str] = [], outputs: list[str] = [], param: nndeploy.base.Param = None):
    op = _C.op.create_op(device_type, name, op_type, inputs, outputs)
    if op is not None:
        op.set_param(param)
    return op


