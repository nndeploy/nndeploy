import torch
import json



# parameters/buffers run info
def get_parameters_buffers_run_info(model: torch.nn.Module, output_file: str = "parameters_buffers_run_info.json"):
    parameters_info = []
    buffers_info = []
    
    for name, param in model.named_parameters():
        parameters_info.append({
            "name": name,
            "shape": list(param.shape),
            "requires_grad": param.requires_grad,
            "device": str(param.device),
            "dtype": str(param.dtype),
            "numel": param.numel()
        })
    
    for name, buffer in model.named_buffers():
        buffers_info.append({
            "name": name,
            "shape": list(buffer.shape),
            "device": str(buffer.device),
            "dtype": str(buffer.dtype),
            "numel": buffer.numel()
        })
    
    run_info = {
        "parameters_run_info": {
            "parameters": parameters_info,
            "buffers": buffers_info,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_buffers": sum(b.numel() for b in model.buffers())
        }
    }
    
    # 写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)
    
    print(f"parameters and buffers run info saved to: {output_file}")
    return run_info

# activation run info
def get_activation_run_info(model: torch.nn.Module, output_file: str = "activation_run_info.json"):       
    def get_activation_post_hook(module, input, output):
        activation_info = []
        # for i, input_tensor in enumerate(input):
        #     activation_info.append({
        #         "name": module.__class__.__name__,
        #         "index": i,
        #         # "input_shape": input_tensor.shape,
        #         "input_device": input_tensor.device,
        #         "input_dtype": input_tensor.dtype
        #     })
        #     print(activation_info[-1])
        for i, output_tensor in enumerate(output):
            activation_info.append({
                "name": module.__class__.__name__,
                "index": i,
                "output_shape": output_tensor.shape,
                "output_device": output_tensor.device,
                "output_dtype": output_tensor.dtype
            })
            print(activation_info[-1])
    for module in model.modules():
        module.register_forward_hook(get_activation_post_hook)

# memory run info

# memory bandwidth run info

# FLOPs run info
import thop
def get_flops_run_info(model: torch.nn.Module, input_shape: tuple, output_file: str = "flops_run_info.json"):
    flops, _ = thop.profile(model, inputs=(input_shape,))
    return flops / (1000 * 1000 * 1000)

# FLOPS run info