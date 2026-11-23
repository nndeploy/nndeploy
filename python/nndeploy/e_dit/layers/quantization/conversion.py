"""Convert explicit Linear targets; selection and fusion policies belong to callers."""
import time
import torch
from torch import nn


def convert_linears(model, names, *, scope='selected', execution_device=None, mode='int8_w8a8_native'):
    from nndeploy.e_dit.layers.quantization.int8 import NativeInt8Linear
    if mode not in ('int8_w8a8_native', 'fp8_w8a8_native', 'fp8_w8a8_tensorwise', 'fp8_w8a8_static'):
        raise ValueError('unsupported quantization mode')
    from nndeploy.e_dit.layers.quantization.fp8 import NativeFp8Linear, NativeTensorwiseFp8Linear, NativeStaticFp8Linear
    linear_class = {'int8_w8a8_native': NativeInt8Linear,
                    'fp8_w8a8_native': NativeFp8Linear,
                    'fp8_w8a8_tensorwise': NativeTensorwiseFp8Linear,
                    'fp8_w8a8_static': NativeStaticFp8Linear}[mode]
    existing = getattr(model, '_quantization_report', None)
    if existing is not None:
        if existing['scope'] != scope or existing['mode'] != mode or existing['selected_names'] != list(names):
            raise ValueError('model already quantized with a different scope or mode')
        return existing
    if (getattr(model, '_layerwise_offload_manager', None) is not None
            or getattr(model, 'layerwise_offload_managers', None)
            or hasattr(model, '_block_compile_report')):
        raise ValueError('quantize before registering offload or compile')
    if getattr(model,'peft_config',None):
        raise ValueError('W8A8 conversion with LoRA adapters is not supported')
    names = list(names)
    if not names or len(names) != len(set(names)):
        raise ValueError('quantization requires nonempty unique module names')
    for name in names:
        module = model.get_submodule(name)
        if not isinstance(module, nn.Linear) or module.in_features%32 or module.out_features%32:
            raise ValueError(f'unsupported W8A8 target: {name}')
        if module.weight.dtype != torch.bfloat16 or module.weight.device.type not in ('cuda', 'cpu'):
            raise ValueError(f'W8A8 target must be CPU/CUDA BF16: {name}')
    device = model.get_submodule(names[0]).weight.device
    source_device = device
    device = torch.device(execution_device) if execution_device is not None else device
    if device.type != 'cuda':
        raise ValueError('CPU quantization needs a CUDA execution_device')
    if torch.cuda.get_device_capability(device) < (8,0):
        raise ValueError('this INT8 implementation requires CUDA capability >= 8.0')
    # Execute the real backend before mutating any model layer.
    if mode.startswith('fp8_'):
        if torch.cuda.get_device_capability(device) < (8, 9):
            raise ValueError('FP8 requires CUDA capability >= 8.9')
        probe = torch.zeros((32, 32), device=device, dtype=torch.float8_e4m3fn)
        unit = torch.ones((), device=device)
        torch._scaled_mm(probe, probe.t(), scale_a=unit, scale_b=unit,
                         out_dtype=(torch.bfloat16 if linear_class.tensorwise else torch.float32),
                         use_fast_accum=linear_class.use_fast_accum)
    else:
        probe = torch.zeros((32,32), device=device,dtype=torch.int8)
        torch._int_mm(probe, probe.t())
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    original_bytes = quantized_bytes = 0
    originals = []
    try:
        for name in names:
            original = model.get_submodule(name)
            original_bytes += sum(t.numel()*t.element_size() for t in original.parameters())
            # CPU and CUDA FP32 division can round scales differently; at FP8 ties
            # this changes many packed weights. Always convert on the execution
            # GPU, one layer at a time, then restore the requested storage location.
            # Offload changes residency, not the quantized model's numerical path.
            replacement = linear_class.from_linear(original, execution_device=device).to(original.weight.device)
            quantized_bytes += sum(t.numel()*t.element_size() for t in replacement.buffers())
            parent, _, child = name.rpartition('.')
            parent_module = model.get_submodule(parent)
            originals.append((parent_module, child, original))
            setattr(parent_module, child, replacement)
    except BaseException:
        for parent, child, original in reversed(originals):
            setattr(parent, child, original)
        raise
    torch.cuda.synchronize(device)
    report = dict(mode=mode,scope=scope,quantized_count=len(names),
                  selected_names=names,source_linear_bytes=original_bytes,
                  quantized_linear_bytes=quantized_bytes,bf16_duplicate_weight_count=0,
                  backend=('torch._int_mm / triton fused expansion GEMM' if mode == 'int8_w8a8_native'
                           else 'torch._scaled_mm E4M3 scaled BF16 output with bias, use_fast_accum=True' if mode == 'fp8_w8a8_static'
                           else 'torch._scaled_mm E4M3 scaled BF16 output with bias' if mode == 'fp8_w8a8_tensorwise'
                           else 'torch._scaled_mm E4M3 FP32 output, use_fast_accum=False') + (' + triton dynamic tensor quantization' if mode == 'fp8_w8a8_tensorwise'
                                else ' + triton static activation quantization' if mode == 'fp8_w8a8_static'
                                else ' + triton quantization/epilogue'),
                  scale_granularity=('fixed-tensor/tensor' if mode == 'fp8_w8a8_static'
                                     else 'tensor/tensor' if mode == 'fp8_w8a8_tensorwise' else 'token/channel'),
                  activation_scale=0.125 if mode == 'fp8_w8a8_static' else None,
                  activation_saturation=56. if mode == 'fp8_w8a8_static' else None,
                  use_fast_accum=mode == 'fp8_w8a8_static',
                  fused_expansion_min_rows=1024 if mode == 'int8_w8a8_native' else None,
                  fused_gelu_count=0,
                  gelu_lut_bytes_per_module=0,
                  conversion_seconds=time.perf_counter()-started, conversion_device=str(device),
                  storage_device=str(source_device))
    model._quantization_report = report
    return report


def runtime_report(model):
    base = getattr(model,'_quantization_report',None)
    if base is None:
        return dict(mode='none',runtime_call_count=0)
    from nndeploy.e_dit.layers.quantization.int8 import NativeInt8Linear
    from nndeploy.e_dit.layers.quantization.fp8 import NativeFp8Linear
    modules = [m for m in model.modules() if isinstance(m, (NativeInt8Linear, NativeFp8Linear))]
    return dict(base,runtime_call_count=sum(m.calls for m in modules),
                runtime_counts_include_compiled=False,
                fused_gemm_call_count=sum(getattr(m, "fused_calls", 0) for m in modules),
                fused_gelu_call_count=sum(getattr(m, "gelu_calls", 0) for m in modules),
                executed_module_count=sum(m.calls>0 for m in modules), fallback_count=0)
