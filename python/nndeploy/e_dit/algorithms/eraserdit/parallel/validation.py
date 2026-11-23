"""EraserDiT-specific NCCL capability and numerical alignment contracts."""
from nndeploy.e_dit.config.dit_parallel import resolve_dit_topology

def validate_nccl_dit(args, batch=None):
    config = getattr(args, 'pipeline_config', None)
    if getattr(config, 'dit_parallel_backend', 'peer') not in ('peer', 'nccl'):
        raise ValueError('dit_parallel_backend must be peer or nccl')
    if getattr(config, 'dit_parallel_backend', 'peer') != 'nccl':
        if getattr(config, 'sp_linear_mode', 'reference') == 'aligned':
            raise ValueError('aligned SP requires resident NCCL Ulysses SP2/4')
        if any(getattr(config, name, 1) != 1 for name in
               ('tp_degree', 'ring_degree', 'dit_fsdp_shard_degree', 'dit_fsdp_replicate_degree')) or getattr(config, 'ulysses_degree', None) is not None:
            raise ValueError('explicit TP/USP/FSDP degrees require --dit-parallel-backend nccl')
        return None
    topology = resolve_dit_topology(config)
    import os
    chunk_policy = os.environ.get('NNDEPLOY_DIT_ERASERDIT_ULYSSES_HEAD_CHUNKS', '1')
    try:
        head_chunks = 2 if chunk_policy == 'auto' else int(chunk_policy)
    except ValueError as error:
        raise ValueError('NNDEPLOY_DIT_ERASERDIT_ULYSSES_HEAD_CHUNKS must be auto, 1, 2 or 4') from error
    if head_chunks not in (1, 2, 4):
        raise ValueError('NNDEPLOY_DIT_ERASERDIT_ULYSSES_HEAD_CHUNKS must be auto, 1, 2 or 4')
    output_overlap = os.environ.get('NNDEPLOY_DIT_ERASERDIT_ULYSSES_OUTPUT_OVERLAP', '0')
    if output_overlap not in ('0', '1'):
        raise ValueError('NNDEPLOY_DIT_ERASERDIT_ULYSSES_OUTPUT_OVERLAP must be 0 or 1')
    if output_overlap == '1' and head_chunks == 1:
        raise ValueError('output overlap requires Ulysses head chunks 2 or 4')
    if head_chunks > 1 and (topology.ulysses not in (2, 4) or topology.ring != 1
            or topology.tp != 1 or topology.replicas != 1
            or config.dit_fsdp_shard_degree * config.dit_fsdp_replicate_degree != 1
            or os.environ.get('NNDEPLOY_DIT_ERASERDIT_NCCL_PACKING', 'reference') != 'direct'):
        raise ValueError('Ulysses head chunks require resident SP2/4, direct packing, no Ring/TP/FSDP')
    import torch
    if args.resolve_component_dtype('transformer') is not torch.bfloat16:
        raise ValueError('NCCL DiT currently requires bf16 model precision')
    if config.sp_linear_mode not in ('reference', 'sharded', 'aligned'):
        raise ValueError('sp_linear_mode must be reference, sharded or aligned')
    if config.sp_linear_mode == 'aligned' and (
            topology.ulysses not in (2, 4) or topology.ring != 1 or topology.tp != 1
            or topology.replicas != 1
            or config.dit_fsdp_shard_degree * config.dit_fsdp_replicate_degree != 1):
        raise ValueError('aligned SP requires resident NCCL Ulysses SP2/4 without TP/Ring/FSDP')
    if config.tp_linear_mode not in ('reference', 'sharded', 'aligned'):
        raise ValueError('tp_linear_mode must be reference, sharded or aligned')
    if type(config.vae_degree) is not int or config.vae_degree not in (1, 2, 4):
        raise ValueError('vae_degree must be 1, 2 or 4')
    tile, stride = config.vae_tile_size, config.vae_tile_stride
    if tile < 64 or stride < 32 or stride >= tile or tile % 32 or stride % 32:
        raise ValueError('VAE tile and stride must be multiples of 32; 32 <= stride < tile')
    if config.ring_attention_mode not in ('reference', 'online', 'streaming'):
        raise ValueError('ring_attention_mode must be reference, online or streaming')
    if args.dit_cpu_offload or args.dit_layerwise_offload:
        raise ValueError('NCCL DiT workers currently require resident or FSDP-sharded weights')
    from nndeploy.e_dit.algorithms.eraserdit.optimizations.compile_config import normalize_compile_components
    if args.enable_torch_compile or normalize_compile_components(getattr(args, 'compile_components', ())):
        raise ValueError('NCCL quality-alignment path requires compile off and unquantized DiT')
    if args.transformer_quantization != 'none' or args.attention_backend != 'sdpa':
        validate_nccl_approximate(args, topology)
    if args.operator_fusion_backend != 'disabled':
        validate_nccl_fusion(args, topology)
    if config.cfg_parallel_device is not None:
        raise ValueError('use cfg_degree with NCCL DiT workers')
    if batch is not None:
        get = batch.get if isinstance(batch, dict) else lambda key, default: getattr(batch, key, default)
        if get('transformer_cache_mode', 'off') != 'off':
            validate_nccl_text_cache(config, topology)
        if get('cache_text_projections', False):
            validate_nccl_text_cache(config, topology)
    return topology


def validate_nccl_text_cache(config, topology):
    """Text K/V reuse is currently validated only for resident CFG/Ulysses."""
    if (topology.tp != 1 or topology.ring != 1 or topology.replicas != 1
            or config.dit_fsdp_shard_degree * config.dit_fsdp_replicate_degree != 1):
        raise ValueError('NCCL text projection cache requires resident CFG/Ulysses without TP, Ring or FSDP')


def validate_nccl_fusion(args, topology):
    config = args.pipeline_config
    if (topology.tp != 1 or topology.ring != 1 or topology.replicas != 1
            or topology.sp not in (1, 2, 4)
            or config.dit_fsdp_shard_degree * config.dit_fsdp_replicate_degree != 1):
        raise ValueError('NCCL operator fusion requires resident CFG/Ulysses with SP1/2/4')
    if args.operator_fusion_backend not in ('auto', 'triton'):
        raise ValueError('NCCL operator fusion backend must be disabled, auto or triton')
    ops = args.operator_fusion_ops
    if ops is not None:
        ops = tuple(s.strip() for s in ops.split(',') if s.strip()) if isinstance(ops, str) else tuple(ops)
        if not ops or set(ops) - {'qk_rmsnorm_rope', 'rmsnorm_adaln',
                                  'qk_rmsnorm_rope_native', 'rmsnorm_adaln_native',
                                  'qk_rmsnorm_rope_fast', 'rmsnorm_adaln_fast', 'gated_residual'}:
            raise ValueError('unsupported NCCL operator fusion operation')
        if set(ops) & {'qk_rmsnorm_rope_fast', 'rmsnorm_adaln_fast', 'gated_residual'}:
            validate_nccl_approximate(args, topology)


def validate_nccl_approximate(args, topology):
    """Explicit approximate path; keep TP/Ring/FSDP and other packing modes closed."""
    config = args.pipeline_config
    if (topology.tp != 1 or topology.ring != 1 or topology.replicas != 1
            or topology.sp not in (1, 2, 4)
            or config.dit_fsdp_shard_degree * config.dit_fsdp_replicate_degree != 1):
        raise ValueError('NCCL approximate path requires resident CFG/Ulysses SP1/2/4')
    if args.attention_backend not in ('sdpa', 'sage_fp8'):
        raise ValueError('NCCL approximate path supports SDPA or Sage FP8')
    if args.transformer_quantization != 'none' and (
            args.transformer_quantization != 'fp8_w8a8_static' or config.quantization_scope != 'ffn'):
        raise ValueError('NCCL quantization supports static FP8 FFN only')
