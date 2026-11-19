"""Fail unsupported migrated-memory combinations before loading or execution."""
import torch


def validate_memory_config(args, batch=None):
    from nndeploy.e_dit.algorithms.eraserdit.parallel.validation import validate_nccl_dit
    validate_nccl_dit(args, batch)
    from nndeploy.e_dit.algorithms.eraserdit.optimizations.compile_config import validate_transformer_compile
    validate_transformer_compile(args, batch)
    from nndeploy.e_dit.algorithms.eraserdit.optimizations.compile_config import validate_component_compile
    validate_component_compile(args)
    memory = args.resolve_resource_policy()
    if memory.enabled and (torch.device(args.device).type != 'cuda' or not torch.cuda.is_available()):
        raise ValueError('SGLang offload requires an available CUDA device')
    if args.use_fsdp_inference:
        raise ValueError('DiT FSDP inference is not supported by this single-GPU integration; T5 CPU offload uses its own FSDP mesh')
    if memory.enabled:
        config = args.pipeline_config
        dit_parallel = max(getattr(config, k, 1) for k in ('sp_degree', 'cfg_degree')) > 1
        if ((dit_parallel and (memory.dit_cpu_offload or memory.dit_layerwise_offload))
                or getattr(config, 'vae_degree', 1) > 1
                or getattr(config, 'cfg_parallel_device', None) is not None):
            raise ValueError('SGLang offload currently requires single-GPU EraserDiT')
        # EraserDiT compilation covers only local FFNs, with per-layer warmup
        # residency and CUDA graphs disabled. Transfer hooks remain eager.
        # Quantization converts one layer on the execution GPU and returns it
        # to its original storage device before placement hooks are registered.
        # Its
        # packed weights, scales and per-block GELU tables are ordinary buffers
        # tracked by both component and layerwise offload managers.
    if batch is not None and args.dit_layerwise_offload:
        if getattr(batch, 'transformer_cache_mode', 'off') not in ('off', 'teacache'):
            raise ValueError('cache_dit cannot be combined with SGLang cyclic layerwise offload')
