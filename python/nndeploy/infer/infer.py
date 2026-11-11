import nndeploy._nndeploy_internal as _C


import nndeploy.base
import nndeploy.device
import nndeploy.op
import nndeploy.inference
import nndeploy.dag


class Infer(_C.infer.Infer):
    def __init__(self, name: str, inputs=None, outputs=None, type=None):
        args = [name]
        if inputs is not None or outputs is not None:
            args.extend((inputs or [], outputs or []))
        if type is not None:
            args.append(type)
        super().__init__(*args)

    def set_input_name(self, name, index=0):
        return super().set_input_name(name, index)

    def set_output_name(self, name, index=0):
        return super().set_output_name(name, index)

    def set_input_names(self, names):
        return super().set_input_names(names)

    def set_output_names(self, names):
        return super().set_output_names(names)

    def set_inference_type(self, inference_type):
        return super().set_inference_type(inference_type)

    def set_param(self, param):
        return super().set_param(param)

    def get_param(self):
        return super().get_param()

    def init(self):
        return super().init()

    def deinit(self):
        return super().deinit()

    def get_memory_size(self):
        return super().get_memory_size()

    def set_memory(self, buffer):
        return super().set_memory(buffer)

    def run(self):
        return super().run()

    def get_inference(self):
        return super().get_inference()
        
        
