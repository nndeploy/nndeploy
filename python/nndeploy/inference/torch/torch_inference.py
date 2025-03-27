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
    
    def set_param(self, param):
        super().set_param(param)
        
    def get_param(self):
        return super().get_param()
        
    def set_stream(self, stream):
        super().set_stream(stream)
          
    def get_stream(self):
        inner_stream = super().get_stream()
        return nndeploy.device.Stream(inner_stream)
          
    def init(self):
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
      
    def get_num_of_input_tensor(self):
        # 获取模型参数
        param = self.get_param()
        if param.get_input_num() == 0:
            nn_model = param.nn_model
            # 获取forward方法的参数个数(排除self)
            import inspect
            signature = inspect.signature(nn_model.forward)
            # 减去self参数
            num_params = len(signature.parameters) - 1
        else:
            num_params = param.input_num_
        
        return num_params
      
    def get_num_of_output_tensor(self):
      param = self.get_param()
      if param.get_output_num() == 0:
          nn_model = param.nn_model
          # 获取forward方法的返回值个数
          import inspect
          signature = inspect.signature(nn_model.forward)
          num_params = len(signature.return_annotations)
      else:
          num_params = param.output_num_
          
      return num_params
    
    def get_input_name(self, i):
        param = self.get_param()
        input_name = ""
        if len(param.get_input_name()) <= i:
            nn_model = param.nn_model
            # 获取forward方法的参数个数(排除self)
            import inspect
            signature = inspect.signature(nn_model.forward)
            # 减去self参数
            input_name = signature.parameters[i].name
        else:
            input_name = param.get_input_name()[i]
          
        return input_name
    
    def get_output_name(self, i):
        param = self.get_param()
        output_name = ""
        if len(param.get_output_name()) <= i:
            nn_model = param.nn_model
            # 获取forward方法的返回值个数
            import inspect
            signature = inspect.signature(nn_model.forward)
            output_name = signature.return_annotations[i].name
        else:
            output_name = param.get_output_name()[i]
          
        return output_name
    
    def get_all_input_tensor_name(self):
        param = self.get_param()
        input_name = param.get_input_name()
        if len(input_name) == 0:
            nn_model = param.nn_model
            # 获取forward方法的参数个数(排除self)
            import inspect
            signature = inspect.signature(nn_model.forward)
            input_name = [signature.parameters[i].name for i in range(len(signature.parameters) - 1)]
        return input_name

    def get_all_output_tensor_name(self):
        param = self.get_param()
        output_name = param.get_output_name()
        if len(output_name) == 0:
            nn_model = param.nn_model
            # 获取forward方法的返回值个数
            import inspect
            signature = inspect.signature(nn_model.forward)
            output_name = [signature.return_annotations[i].name for i in range(len(signature.return_annotations))]
        return output_name
    
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
        self.output_tensors = self.nn_model(*inputs)
        self._index = 0
        return nndeploy.base.Status("ok")
        
    def get_output(self):
        return self.output_tensors
      
    def get_output_tensor_after_run(self, name, device_type, is_copy, data_format):
        return nndeploy.base.Status("ok")


class InferenceCreator(_C.inference.InferenceCreator):
    def __init__(self):
        super().__init__()

    def create_inference(self, type):
        return TorchInference(type)


register_inference_creator(nndeploy.base.InferenceType.Torch, TorchInferenceCreator())