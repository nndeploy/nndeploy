import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.op
import nndeploy.dag


ConvertToParam = _C.preprocess.ConvertToParam

class ConvertTo(_C.preprocess.ConvertTo):
    def __init__(self, name: str, inputs: list[nndeploy.dag.Edge] = [], outputs: list[nndeploy.dag.Edge] = []):
        if inputs is [] and outputs is []:
            super().__init__(name)
        else:
            super().__init__(name, inputs, outputs)

    def run(self):
        return super().run()
      

class CvtNormTrans(_C.preprocess.CvtNormTrans):
    def __init__(self, name: str, inputs: list[nndeploy.dag.Edge] = [], outputs: list[nndeploy.dag.Edge] = []):
        if inputs is [] and outputs is []:
            super().__init__(name)
        else:
            super().__init__(name, inputs, outputs)

    def run(self):
        return super().run()
      

class CvtResizeCropNormTrans(_C.preprocess.CvtResizeCropNormTrans):
    def __init__(self, name: str, inputs: list[nndeploy.dag.Edge] = [], outputs: list[nndeploy.dag.Edge] = []):
        if inputs is [] and outputs is []:
            super().__init__(name)
        else:
            super().__init__(name, inputs, outputs)

    def run(self):
        return super().run()
    

class CvtResizePadNormTrans(_C.preprocess.CvtResizePadNormTrans):
    def __init__(self, name: str, inputs: list[nndeploy.dag.Edge] = [], outputs: list[nndeploy.dag.Edge] = []):
        if inputs is [] and outputs is []:
            super().__init__(name)
        else:
            super().__init__(name, inputs, outputs)

    def run(self):
        return super().run()
      
      
class CvtResizeNormTrans(_C.preprocess.CvtResizeNormTrans):
    def __init__(self, name: str, inputs: list[nndeploy.dag.Edge] = [], outputs: list[nndeploy.dag.Edge] = []):
        if inputs is [] and outputs is []:
            super().__init__(name)
        else:
            super().__init__(name, inputs, outputs)

    def run(self):
        return super().run()
      
      
class BatchPreprocess(_C.preprocess.BatchPreprocess):
    def __init__(self, name: str, inputs: list[nndeploy.dag.Edge] = [], outputs: list[nndeploy.dag.Edge] = []):
        if inputs is [] and outputs is []:
            super().__init__(name)
        else:
            super().__init__(name, inputs, outputs)

    def set_node_key(self, key: str):
        return super().set_node_key(key)

    def set_data_format(self, data_format):
        return super().set_data_format(data_format)

    def get_data_format(self):
        return super().get_data_format()

    def set_param(self, param):
        return super().set_param(param)

    def get_param(self):
        return super().get_param()

    def run(self):
        return super().run()

    def serialize(self):
        return super().serialize()

    def deserialize(self, json_str):
        return super().deserialize(json_str)
