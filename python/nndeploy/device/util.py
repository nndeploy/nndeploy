import torch
import platform

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
  