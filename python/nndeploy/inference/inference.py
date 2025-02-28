
import nndeploy._nndeploy_internal as _C


import nndeploy.base
import nndeploy.device
import nndeploy.ir
import nndeploy.op

from .inference_param import InferenceParam, InferenceParamCreator, register_inference_param_creator, create_inference_param


# python3 nndeploy/inference/inference.py


class Inference(_C.inference.Inference):
    def __init__(self, type):
        super().__init__(type)
        
    def get_inference_type(self):
        return super().get_inference_type()
    
    def set_param(self, param):
        return super().set_param(param)
    
    def get_param(self):
        return super().get_param()
    
    def get_device_type(self):
        return super().get_device_type()
    
    def set_stream(self, stream):
        return super().set_stream(stream)
    
    def get_stream(self):
        return super().get_stream()
    
    def init(self):
        return super().init()
    
    def deinit(self):
        return super().deinit()
    
    def get_min_shape(self):
        return super().get_min_shape()
    
    def get_opt_shape(self):
        return super().get_opt_shape()
    
    def get_max_shape(self):
        return super().get_max_shape()
    
    def reshape(self, shape_map):
        return super().reshape(shape_map)
    
    def get_memory_size(self):
        return super().get_memory_size()
    
    def set_memory(self, buffer):
        return super().set_memory(buffer)
    
    def get_gflops(self):
        return super().get_gflops()
    
    def is_batch(self):
        return super().is_batch()
    
    def is_share_context(self):
        return super().is_share_context()
    
    def is_share_stream(self):
        return super().is_share_stream()
    
    def is_input_dynamic(self):
        return super().is_input_dynamic()
    
    def is_output_dynamic(self):
        return super().is_output_dynamic()
    
    def can_op_input(self):
        return super().can_op_input()
    
    def can_op_output(self):
        return super().can_op_output()
    
    def get_num_of_input_tensor(self):
        return super().get_num_of_input_tensor()
    
    def get_num_of_output_tensor(self):
        return super().get_num_of_output_tensor()
    
    def get_input_name(self, i=0):
        return super().get_input_name(i)
    
    def get_output_name(self, i=0):
        return super().get_output_name(i)
    
    def get_all_input_tensor_name(self):
        return super().get_all_input_tensor_name()
    
    def get_all_output_tensor_name(self):
        return super().get_all_output_tensor_name()
    
    def get_input_shape(self, name):
        return super().get_input_shape(name)
    
    def get_all_input_shape(self):
        return super().get_all_input_shape()
    
    def get_input_tensor_desc(self, name):
        return super().get_input_tensor_desc(name)
    
    def get_output_tensor_desc(self, name):
        return super().get_output_tensor_desc(name)
    
    def get_input_tensor_align_desc(self, name):
        return super().get_input_tensor_align_desc(name)
    
    def get_output_tensor_align_desc(self, name):
        return super().get_output_tensor_align_desc(name)
    
    def get_all_input_tensor_map(self):
        return super().get_all_input_tensor_map()
    
    def get_all_output_tensor_map(self):
        return super().get_all_output_tensor_map()
    
    def get_all_input_tensor_vector(self):
        return super().get_all_input_tensor_vector()
    
    def get_all_output_tensor_vector(self):
        return super().get_all_output_tensor_vector()
    
    def get_input_tensor(self, name):
        return super().get_input_tensor(name)
    
    def get_output_tensor(self, name):
        return super().get_output_tensor(name)
    
    def set_input_tensor(self, name, input_tensor):
        return super().set_input_tensor(name, input_tensor)
    
    def run(self):
        return super().run()
    
    def get_output_tensor_after_run(self, name, device_type, is_copy, data_format=nndeploy.base.DataFormat.Auto):
        return super().get_output_tensor_after_run(name, device_type, is_copy, data_format)


class InferenceCreator(_C.inference.InferenceCreator):
    def __init__(self):
        super().__init__()

    def create_inference(self, type):
        return super().create_inference(type)



def register_inference_creator(type, creator):
    return _C.inference.register_inference_creator(type, creator)



def create_inference(type):
    return _C.inference.create_inference(type)



