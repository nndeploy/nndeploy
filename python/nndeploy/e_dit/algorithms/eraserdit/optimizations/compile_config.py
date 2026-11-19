"""Contracts for EraserDiT's FFN and whole-transformer compilation scopes."""


def validate_transformer_compile(args, batch=None):
    if not getattr(args, 'enable_torch_compile', False):
        return
    if getattr(args, 'torch_compile_scope', 'ffn') != 'transformer':
        return
    if any(getattr(args, name, False) for name in
           ('dit_cpu_offload', 'dit_layerwise_offload', 'use_fsdp_inference')):
        raise ValueError('transformer compile requires resident DiT weights; disable DiT offload')
    config = args.pipeline_config
    if ((getattr(config, 'sp_degree', 1) or 1) > 1
            or (getattr(args, 'sp_degree', 1) or 1) > 1):
        raise ValueError('transformer compile currently requires SP1; CFG1/CFG2 are supported')
    if getattr(config, 'cfg_parallel_device', None) is not None:
        raise ValueError('transformer compile requires --cfg-degree 2 instead of cfg_parallel_device')
    if any((getattr(owner, name, 1) or 1) not in (1, 2)
           for owner, name in ((config, 'cfg_degree'), (args, 'cfg_parallel_degree'))):
        raise ValueError('transformer compile supports CFG1/CFG2 only')
    if args.transformer_quantization != 'none':
        raise ValueError('transformer compile currently requires unquantized DiT')
    if args.operator_fusion_backend != 'disabled':
        raise ValueError('transformer compile requires --operator-fusion-backend disabled; Inductor performs fusion')
    if args.attention_backend != 'sdpa':
        raise ValueError('transformer compile currently requires explicit sdpa attention')
    if batch is not None:
        get = batch.get if isinstance(batch, dict) else lambda key, default: getattr(batch, key, default)
        if get('transformer_cache_mode', 'off') != 'off' or get('cache_text_projections', None):
            raise ValueError('transformer compile requires residual and text projection caches off')


COMPILABLE_COMPONENTS = ('text_encoder', 'vae_encoder', 'vae_decoder')


def normalize_compile_components(value):
    if value is None:
        return ()
    if isinstance(value, str):
        value = value.split(',') if value.strip() else ()
    if not isinstance(value, (tuple, list)):
        raise ValueError('compile_components must be a comma-separated string or sequence')
    values = tuple(dict.fromkeys(str(item).strip() for item in value))
    if any(item not in COMPILABLE_COMPONENTS for item in values):
        raise ValueError('compile_components supports only: ' + ','.join(COMPILABLE_COMPONENTS))
    return values


def validate_component_compile(args):
    selected = normalize_compile_components(getattr(args, 'compile_components', ()))
    if 'text_encoder' in selected and getattr(args, 'text_encoder_cpu_offload', False):
        raise ValueError('text_encoder compile requires --no-text-encoder-cpu-offload (FSDP hooks unsupported)')
    if any(name.startswith('vae_') for name in selected):
        config = args.pipeline_config
        if (getattr(config, 'vae_degree', 1) or 1) > 1 or (getattr(args, 'vae_parallel_degree', 1) or 1) > 1:
            raise ValueError('VAE compile requires single-GPU VAE')
    return selected
