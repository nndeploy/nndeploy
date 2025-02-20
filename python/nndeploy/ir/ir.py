
import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
from .op_param import OpType, register_op_param_creator, create_op_param


class OpDesc(_C.ir.OpDesc):
    def __init__(self, name: str = "", op_type: _C.ir.OpType = _C.ir.OpType.kOpTypeNone, 
                 inputs: list = [], outputs: list = [], op_param: _C.base.Param = None):
        super().__init__(name, op_type, inputs, outputs, op_param)
        
    @property
    def name(self):
        return self.name_
    
    @name.setter
    def name(self, value):
        self.name_ = value
        
    @property  
    def op_type(self):
        return self.op_type_
    
    @op_type.setter
    def op_type(self, value):
        self.op_type_ = value
        
    @property
    def inputs(self):
        return self.inputs_
    
    @inputs.setter
    def inputs(self, value):
        self.inputs_ = value
        
    @property
    def outputs(self):
        return self.outputs_
    
    @outputs.setter
    def outputs(self, value):
        self.outputs_ = value
        
    @property
    def op_param(self):
        return self.op_param_
    
    @op_param.setter
    def op_param(self, value):
        self.op_param_ = value


class ValueDesc(_C.ir.ValueDesc):
    def __init__(self, name: str = "", data_type: nndeploy.base.DataType = nndeploy.base.DataType(nndeploy.base.DataTypeCode.Fp, 32, 1), 
                 shape = []):
        super().__init__(name, data_type, shape)
        
    @property
    def name(self):
        return self.name_
    
    @name.setter
    def name(self, value):
        self.name_ = value
        
    @property
    def data_type(self):
        return self.data_type_
    
    @data_type.setter
    def data_type(self, value):
        self.data_type_ = value
        
    @property
    def shape(self):
        return self.shape_
    
    @shape.setter
    def shape(self, value):
        self.shape_ = value



class ModelDesc(_C.ir.ModelDesc):
    def __init__(self):
        super().__init__()
        
    @property
    def name(self):
        return self.name_
    
    @name.setter
    def name(self, value):
        self.name_ = value
        
    @property
    def metadata(self):
        return self.metadata_
    
    @metadata.setter
    def metadata(self, value):
        self.metadata_ = value
        
    @property
    def inputs(self):
        return self.inputs_
    
    @inputs.setter
    def inputs(self, value):
        self.inputs_ = value
        
    @property
    def outputs(self):
        return self.outputs_
    
    @outputs.setter
    def outputs(self, value):
        self.outputs_ = value
        
    @property
    def op_descs(self):
        return self.op_descs_
    
    @op_descs.setter
    def op_descs(self, value):
        self.op_descs_ = value
        
    @property
    def values(self):
        return self.values_
    
    @values.setter
    def values(self, value):
        self.values_ = value
        
    @property
    def weights(self):
        return self.weights_
    
    def set_weights(self, weights: dict):
        super().set_weights(weights)
            
    def dump(self):
        return super().dump()
    
    def serialize_structure_to_json(self, *args):
        return super().serialize_structure_to_json(*args)
    
    def deserialize_structure_from_json(self, *args):
        return super().deserialize_structure_from_json(*args)
    
    def serialize_weights_to_safetensors(self, *args):
        return super().serialize_weights_to_safetensors(*args)
    
    def deserialize_weights_from_safetensors(self, *args):
        return super().deserialize_weights_from_safetensors(*args)

