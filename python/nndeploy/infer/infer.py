import nndeploy._nndeploy_internal as _C


import nndeploy.base
import nndeploy.device
import nndeploy.op
import nndeploy.inference
import nndeploy.dag


class Infer(_C.infer.Infer):
    def __init__(self, name: str, inputs: list[nndeploy.dag.Edge] = [], outputs: list[nndeploy.dag.Edge] = [], type: nndeploy.base.InferenceType = None):
        if inputs is [] and outputs is [] and type is None:
            super().__init__(name)
        elif inputs is not [] and outputs is not [] and type is None:
            super().__init__(name, inputs, outputs)
        elif inputs is [] and outputs is [] and type is not None:
            super().__init__(name, type)
        else:
            super().__init__(name, inputs, outputs, type)

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
        
        
