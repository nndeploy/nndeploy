"""EraserDiT target selection, GELU pattern policy and capability checks."""
import torch
from torch import nn
from nndeploy.e_dit.layers.quantization.conversion import runtime_report


def validate_quantization(args, batch=None):
    mode = getattr(args, 'transformer_quantization', 'none')
    if mode == 'none':
        return
    if mode not in ('int8_w8a8_native', 'fp8_w8a8_native', 'fp8_w8a8_tensorwise', 'fp8_w8a8_static'):
        raise ValueError('EraserDiT supports none, int8_w8a8_native, fp8_w8a8_native or fp8_w8a8_tensorwise / fp8_w8a8_static')
    c = args.pipeline_config
    if getattr(args, 'use_fsdp_inference', False):
        raise ValueError('EraserDiT W8A8 cannot wrap FSDP models')
    if (args.operator_fusion_backend != 'disabled' and getattr(c, 'sp_degree', 1) > 1
            and getattr(c, 'dit_parallel_backend', None) != 'nccl'):
        raise ValueError('EraserDiT W8A8 with SP requires operator fusion disabled')
    if c.cfg_parallel_device:
        raise ValueError('use cfg_degree for composable W8A8 CFG')


def selected_names(model, scope):
    suffixes = ['ff.net.0.proj', 'ff.net.2']
    if scope == 'blocks':
        suffixes += ['attn1.to_q','attn1.to_k','attn1.to_v','attn1.to_out.0',
                     'attn2.to_q','attn2.to_out.0']
    elif scope == 'ffn_up':
        suffixes = ['ff.net.0.proj']
    elif scope != 'ffn':
        raise ValueError('quantization scope must be blocks, ffn or ffn_up')
    return [f'transformer_blocks.{i}.{s}' for i in range(len(model.transformer_blocks)) for s in suffixes]


def quantize_transformer(model, scope='blocks', *, execution_device=None, mode='int8_w8a8_native'):
    from nndeploy.e_dit.layers.quantization.conversion import convert_linears
    from nndeploy.e_dit.layers.quantization.gelu import make_gelu_lut, ProjectionBeforeFusedGelu
    from diffusers.models.activations import GELU
    existing = getattr(model, '_quantization_report', None)
    targets = []
    if existing is None and scope in ('blocks', 'ffn'):
        for block in model.transformer_blocks:
            net = block.ff.net
            if (type(net[0]) is GELU and net[0].approximate == 'tanh'
                    and (type(net[1]) is nn.Identity
                         or (type(net[1]) is nn.Dropout and net[1].p == 0))):
                targets.append(net)
    report = convert_linears(model, selected_names(model, scope), scope=scope,
                             execution_device=execution_device, mode=mode)
    if targets:
        lut = make_gelu_lut(torch.device(report['conversion_device']))
        for net in targets:
            net[2].gelu_lut = lut.to(net[2].weight_scale.device, copy=True)
            net[0] = ProjectionBeforeFusedGelu(net[0].proj)
        report.update(fused_gelu_count=len(targets), gelu_lut_bytes_per_module=131072)
        report['backend'] += ' + exact BF16 GELU lookup fusion'
    model._eraserdit_quantization_report = report
    if mode == 'int8_w8a8_native':
        model._eraserdit_int8_report = report
    return report
