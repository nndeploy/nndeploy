import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.ir
import nndeploy.op
from ..inference import Inference, InferenceCreator, register_inference_creator


class TorchInference(Inference):
    def __init__(self, inference_type):
        super().__init__(inference_type)
        self.input_tensors_ = {}
        self.output_tensors = None
        self._index = 0
        self._torch_param = None
    
    def set_param(self, param):
        status = super().set_param(param)
        if status:
            # Native copyTo only copies C++ fields, not Python's nn_model.
            self._torch_param = param
            self.output_tensors = None
        return status

    def get_param(self):
        if self._torch_param is not None:
            return self._torch_param
        return super().get_param()
        
    def set_stream(self, stream):
        super().set_stream(stream)
          
    def get_stream(self):
        inner_stream = super().get_stream()
        return nndeploy.device.Stream(inner_stream)
          
    def init(self):
        self._model()
        return nndeploy.base.Status("ok")
    
    def deinit(self):
        return nndeploy.base.Status("ok")
    
    def reshape(self, shape_map):
        return nndeploy.base.Status("ok")
    
    def get_memory_size(self):
        # 获取模型参数
        param = self.get_param()
        nn_model = param.nn_model
        
        # 计算模型参数占用的内存
        param_memory = sum(p.numel() * p.element_size() for p in nn_model.parameters())
        
        # 计算模型buffer占用的内存
        buffer_memory = sum(b.numel() * b.element_size() for b in nn_model.buffers())
        
        # 返回总内存大小(字节)
        return param_memory + buffer_memory
    
    def set_memory(self, buffer):
        # PyTorch的nn.Module使用内部内存管理机制，不需要外部设置内存
        # 忽略传入的buffer参数，直接返回成功状态
        return nndeploy.base.Status("ok")
      
    def get_gflops(self):
        import torch
        # 获取模型参数
        param = self.get_param()
        nn_model = param.nn_model
        
        # 使用thop库计算FLOPs
        from thop import profile
        
        # 创建一个示例输入
        input_shape = self.get_input_shape(self.get_input_name())
        dummy_input = torch.randn(input_shape).to(nn_model.device)
        
        # 计算FLOPs
        flops, _ = profile(nn_model, inputs=(dummy_input,))
        
        # 转换为GFLOPs
        return flops / (1000 * 1000 * 1000)
      
    def is_batch(self):
        return True
      
    def is_share_context(self):
        return True
      
    def is_input_dynamic(self):
        return True
      
    def is_output_dynamic(self):
        return True
      
    def can_op_input(self):
        return False
    
    def can_op_output(self):
        return False
      
    def _model(self):
        model = getattr(self.get_param(), "nn_model", None)
        if model is None:
            raise ValueError("Torch inference requires param.nn_model to be set")
        return model

    def get_num_of_input_tensor(self):
        return self.get_param().get_input_num() or len(self.get_all_input_tensor_name())

    def get_num_of_output_tensor(self):
        param = self.get_param()
        if param.get_output_num():
            return param.get_output_num()
        if param.get_output_name():
            return len(param.get_output_name())
        if self.output_tensors is not None:
            if isinstance(self.output_tensors, (tuple, list, dict)):
                return len(self.output_tensors)
            return 1
        # Python annotations do not reliably describe runtime output arity.
        raise ValueError("Set output_num/output_name or run the model before querying outputs")

    def get_input_name(self, i=0):
        return self.get_all_input_tensor_name()[i]

    def get_output_name(self, i=0):
        return self.get_all_output_tensor_name()[i]

    def get_all_input_tensor_name(self):
        names = self.get_param().get_input_name()
        if names:
            return names
        import inspect
        parameters = inspect.signature(self._model().forward).parameters.values()
        if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in parameters):
            raise ValueError("Set input_name for models with variadic inputs")
        return [p.name for p in parameters]

    def get_all_output_tensor_name(self):
        names = self.get_param().get_output_name()
        if names:
            return names
        if isinstance(self.output_tensors, dict):
            return list(self.output_tensors)
        return [str(i) for i in range(self.get_num_of_output_tensor())]

    def get_input_shape(self, name):
        # 获取模型参数
        param = self.get_param()
        input_shape = param.get_input_shape(name)
        if len(input_shape) == 0:
            nn_model = param.nn_model
        
            # 尝试从模型中获取输入形状
            # 有些模型可能在第一层定义了输入大小
            try:
                # 检查模型是否有明确的输入大小定义
                if hasattr(nn_model, 'input_size'):
                    input_shape = nn_model.input_size
            except:
                pass
        return input_shape
        
    def get_all_input_shape(self):
        param = self.get_param()
        input_shape = param.get_input_shape()
        if len(input_shape) == 0:
            nn_model = param.nn_model
            # 尝试从模型中获取输入形状
            # 有些模型可能在第一层定义了输入大小
            try:
                # 检查模型是否有明确的输入大小定义
                if hasattr(nn_model, 'input_size'):
                    input_shape.append(nn_model.input_size) 
            except:
                pass
        return input_shape
        
    def get_input_tensor_desc(self, name):
        import torch
        param = self.get_param()
        input_shape = self.get_input_shape(name)
        # 尝试从nn.Module中获取输入数据类型
        data_type = nndeploy.base.DataType("float32")
        try:
            if hasattr(param.nn_model, 'input_dtype'):
                # 如果模型定义了输入数据类型属性
                model_dtype = param.nn_model.input_dtype
                if model_dtype == torch.float16:
                    data_type = nndeploy.base.DataType("float16")
                elif model_dtype == torch.int8:
                    data_type = nndeploy.base.DataType("int8")
                elif model_dtype == torch.int32:
                    data_type = nndeploy.base.DataType("int32")
                else:
                    data_type = nndeploy.base.DataType("float32")
        except:
            pass
        if len(input_shape) == 1:
            data_format = nndeploy.base.DataFormat("N")
        elif len(input_shape) == 2:
            data_format = nndeploy.base.DataFormat("NC")
        elif len(input_shape) == 3:
            data_format = nndeploy.base.DataFormat("NCH")
        elif len(input_shape) == 4:
            data_format = nndeploy.base.DataFormat("NCHW")
        elif len(input_shape) == 5:
            data_format = nndeploy.base.DataFormat("NCDHW")
        else:
            data_format = nndeploy.base.DataFormat("NCHW")
        return nndeploy.device.TensorDesc(input_shape, data_type, data_format)
        
    def get_output_tensor_desc(self, name):
        import torch
        param = self.get_param()
        nn_model = param.nn_model
        
        # 使用trace方法获取输出形状和类型
        try:
            # 获取输入形状
            input_shape = param.get_input_shape()
            inputs = []
            for i in range(len(input_shape)): 
                # 创建一个示例输入张量
                example_input = torch.rand(input_shape[i])
                inputs.append(example_input)
            # 使用trace跟踪模型执行
            traced_model = torch.jit.trace(nn_model, inputs)
            # 执行模型获取输出
            with torch.no_grad():
                output = traced_model(*inputs)
            
            output_shape = list(output.shape)
            if output.dtype == torch.float16:
                data_type = nndeploy.base.DataType("float16")
            elif output.dtype == torch.int8:
                data_type = nndeploy.base.DataType("int8")
            elif output.dtype == torch.int32:
                data_type = nndeploy.base.DataType("int32")
            else:
                data_type = nndeploy.base.DataType("float32")
            # 根据输出形状确定数据格式
            if len(output_shape) == 1:
                data_format = nndeploy.base.DataFormat("N")
            elif len(output_shape) == 2:
                data_format = nndeploy.base.DataFormat("NC")
            elif len(output_shape) == 3:
                data_format = nndeploy.base.DataFormat("NCH")
            elif len(output_shape) == 4:
                data_format = nndeploy.base.DataFormat("NCHW")
            elif len(output_shape) == 5:
                data_format = nndeploy.base.DataFormat("NCDHW")
            else:
                data_format = nndeploy.base.DataFormat("NCHW")
            return nndeploy.device.TensorDesc(output_shape, data_type, data_format)
        except:
            return nndeploy.device.TensorDesc()
          
    def get_input_tensor_align_desc(self, name):
        return self.get_input_tensor_desc(name)
    
    def get_output_tensor_align_desc(self, name):
        return self.get_output_tensor_desc(name)
          
    def set_input(self, tensor):
        name = str(self._index)
        self.input_tensors_[name] = tensor
        self._index += 1
        
        return nndeploy.base.Status("ok")
    
    def run(self):
        inputs = []
        for key, value in self.input_tensors_.items():
            inputs.append(value)
        self.output_tensors = None
        try:
            self.output_tensors = self._model()(*inputs)
        finally:
            self.input_tensors_.clear()
            self._index = 0
        return nndeploy.base.Status("ok")
        
    def get_output(self):
        return self.output_tensors
      
    def get_output_tensor_after_run(self, name, device_type, is_copy, data_format):
        return nndeploy.base.Status("ok")


class TorchInferenceCreator(InferenceCreator):
    def __init__(self):
        super().__init__()

    def create_inference(self, type):
        return TorchInference(type)


_torch_creator = TorchInferenceCreator()
register_inference_creator(nndeploy.base.InferenceType.Torch, _torch_creator)
