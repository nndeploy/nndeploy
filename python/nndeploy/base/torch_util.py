
import torch
import platform
import numpy as np
from typing import Union, Optional, Tuple, List, Any
from PIL import Image

def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """获取torch数据类型对象"""
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
        "double": torch.double,
        "half": torch.half,
    }
    return dtype_map.get(dtype_str, torch.float16)

def get_torch_device(device_str: str = "auto") -> torch.device:
    """获取torch设备对象"""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        else:
            print("CUDA不可用，回退到CPU")
            return torch.device("cpu")
    else:
        return torch.device(device_str)

def torch_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """将torch张量转换为numpy数组"""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.detach().numpy()

def numpy_to_torch_tensor(array: np.ndarray, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """将numpy数组转换为torch张量"""
    tensor = torch.from_numpy(array)
    if dtype is not None:
        tensor = tensor.to(dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor

def pil_to_torch_tensor(image: Image.Image, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """将PIL图像转换为torch张量 (C, H, W)"""
    # 转换为numpy数组
    array = np.array(image)
    
    # 处理不同的图像格式
    if len(array.shape) == 2:  # 灰度图
        array = array[None, :, :]  # 添加通道维度
    elif len(array.shape) == 3:  # RGB图像
        array = array.transpose(2, 0, 1)  # HWC -> CHW
    
    # 归一化到[0, 1]
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    
    tensor = torch.from_numpy(array).to(dtype)
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor

def torch_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """将torch张量转换为PIL图像"""
    # 移动到CPU并转换为numpy
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # 处理批次维度
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # 取第一个样本
    
    # 确保是CHW格式
    if len(tensor.shape) == 3:
        array = tensor.detach().numpy().transpose(1, 2, 0)  # CHW -> HWC
    else:
        array = tensor.detach().numpy()
    
    # 转换到[0, 255]范围
    if array.dtype == np.float32 or array.dtype == np.float64:
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    
    # 处理单通道图像
    if len(array.shape) == 3 and array.shape[2] == 1:
        array = array.squeeze(2)
    
    return Image.fromarray(array)

def get_memory_usage_by_torch(device: torch.device = None) -> dict:
    """获取GPU内存使用情况"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        return {
            "allocated": torch.cuda.memory_allocated(device) / 1024**3,  # GB
            "cached": torch.cuda.memory_reserved(device) / 1024**3,      # GB
            "max_allocated": torch.cuda.max_memory_allocated(device) / 1024**3,  # GB
        }
    else:
        return {"message": "CPU设备不支持内存统计"}

def clear_gpu_cache_by_torch():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def set_random_seed_by_torch(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_torch_model_size(model: torch.nn.Module) -> dict:
    """获取模型大小信息"""
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 估算内存占用（字节）
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    return {
        "total_params": param_count,
        "trainable_params": trainable_count,
        "param_size_mb": param_size / 1024**2,
        "buffer_size_mb": buffer_size / 1024**2,
        "total_size_mb": (param_size + buffer_size) / 1024**2,
    }

def move_to_device(obj: Any, device: torch.device):
    """递归地将对象移动到指定设备"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    elif hasattr(obj, 'to'):
        return obj.to(device)
    else:
        return obj

def safe_load_state_dict_by_torch(model: torch.nn.Module, state_dict: dict, strict: bool = False) -> Tuple[List[str], List[str]]:
    """安全加载模型状态字典"""
    model_keys = set(model.state_dict().keys())
    state_dict_keys = set(state_dict.keys())
    
    missing_keys = list(model_keys - state_dict_keys)
    unexpected_keys = list(state_dict_keys - model_keys)
    
    if not strict:
        # 只加载匹配的键
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        model.load_state_dict(filtered_state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=True)
    
    return missing_keys, unexpected_keys

def batch_torch_tensor(tensors: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    """将不同尺寸的张量批处理（填充到相同尺寸）"""
    if not tensors:
        raise ValueError("张量列表不能为空")
    
    # 获取最大尺寸
    max_dims = []
    for dim in range(len(tensors[0].shape)):
        max_dim = max(tensor.shape[dim] for tensor in tensors)
        max_dims.append(max_dim)
    
    # 填充所有张量到相同尺寸
    padded_tensors = []
    for tensor in tensors:
        pad_sizes = []
        for dim in reversed(range(len(tensor.shape))):
            pad_before = 0
            pad_after = max_dims[dim] - tensor.shape[dim]
            pad_sizes.extend([pad_before, pad_after])
        
        if any(pad_sizes):
            padded = torch.nn.functional.pad(tensor, pad_sizes, value=pad_value)
        else:
            padded = tensor
        padded_tensors.append(padded)
    
    return torch.stack(padded_tensors)

def get_available_device():
    """
    获取可用的设备并按优先级返回
    
    返回:
        selected_device: torch.device, 选择的设备
        device_priority: list, 设备优先级列表
    """
    TENSORRT_AVAILABLE = False
    try:
        import torch_tensorrt
        TENSORRT_AVAILABLE = True
    except ImportError as im:
        print(f"TensorRT is not available: {im}")
    except Exception as e:
        print(f"TensorRT is not available: {e}")

    selected_device = None
    device_priority = []

    if TENSORRT_AVAILABLE and torch.cuda.is_available():
        selected_device = torch.device("cuda")
        device_priority.append("TensorRT+CUDA") 
    elif torch.cuda.is_available():
        selected_device = torch.device("cuda")
        device_priority.append("CUDA")
    elif torch.backends.mps.is_available() and platform.system() == "Darwin":
        selected_device = torch.device("mps")
        device_priority.append("MPS")
    else:
        selected_device = torch.device("cpu")
        device_priority.append("CPU")
        
    return selected_device, device_priority