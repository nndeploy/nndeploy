"""EraserDiT assembly and validation behind the framework model registry."""
from copy import deepcopy
from dataclasses import asdict


from nndeploy.e_dit.algorithms.base import AlgorithmAdapter


class EraserDiTAdapter(AlgorithmAdapter):
    required_components = ("tokenizer", "text_encoder", "vae", "transformer", "scheduler")
    declared_model_index = {
        "tokenizer": ("transformers", "T5Tokenizer"),
        "text_encoder": ("transformers", "T5EncoderModel"),
        "vae": ("diffusers", "AutoencoderKLLTXVideo"),
        "transformer": ("diffusers", "LTXVideoTransformer3DModel"),
        "scheduler": ("diffusers", "FlowMatchEulerDiscreteScheduler"),
    }
    component_architectures = {
        "EraserDiTLTXVideoTransformer3DModel": "nndeploy.e_dit.algorithms.eraserdit.models.transformer:EraserDiTLTXVideoTransformer3DModel",
        "EraserDiTAutoencoderKLLTXVideo": "nndeploy.e_dit.algorithms.eraserdit.models.vae:EraserDiTAutoencoderKLLTXVideo",
    }

    def capabilities(self):
        from nndeploy.e_dit.config.capabilities import AlgorithmCapabilities
        # Combination restrictions remain in the existing topology/compile validators.
        return AlgorithmCapabilities(tasks=('VIDEO_EDIT',), features=(
            'component_offload', 'layerwise_offload', 'compile', 'quantization',
            'fusion', 'sequence_parallel', 'cfg_parallel',
        ))

    def validate_config(self, args):
        from .parallel.validation import validate_nccl_dit
        from .optimizations.compile_config import validate_transformer_compile, validate_component_compile
        validate_nccl_dit(args)
        validate_transformer_compile(args)
        args.compile_components = validate_component_compile(args)

    def validate_loading(self, args):
        from .optimizations.fusion_policy import resolve_operator_fusion_decision
        args.operator_fusion_decision = resolve_operator_fusion_decision(args)
        from .optimizations.quantization import validate_quantization
        from .optimizations.memory_validation import validate_memory_config
        from .parallel.cfg import validate_cfg_parallel
        from .parallel.mesh import resolve_mesh
        validate_quantization(args)
        validate_cfg_parallel(args)
        resolve_mesh(args)
        validate_memory_config(args)

    def prepare_modules(self, modules, args):
        from .optimizations.vae_memory import configure_vae_memory
        configure_vae_memory(modules["vae"], args.pipeline_config.vae_low_memory,
                             args.pipeline_config.vae_chunk_elements)
        policy = args.resolve_resource_policy()
        for name, enabled in (
            ("text_encoder", policy.text_encoder_cpu_offload),
            ("transformer", policy.dit_cpu_offload or policy.dit_layerwise_offload),
            ("vae", policy.vae_cpu_offload),
        ):
            if enabled:
                modules[name].to(device="cpu")
        if args.transformer_quantization != "none" and args.pipeline_config.dit_parallel_backend != "nccl":
            from .optimizations.quantization import quantize_transformer
            report = quantize_transformer(modules["transformer"], args.pipeline_config.quantization_scope,
                                          execution_device=args.device, mode=args.transformer_quantization)
            args.effective_transformer_quantization = report["mode"]
            args.transformer_quantization_report = report

    def compile_modules(self, modules, config):
        from .optimizations.component_compile import ComponentCompileManager
        from nndeploy.e_dit.nodes.compile import resolve_torch_compile_mode
        return ComponentCompileManager(modules, config.compile_components, mode=resolve_torch_compile_mode())

    def initialize_resources(self, resources, args):
        from .resource_hooks import initialize_resources
        initialize_resources(resources, args)

    def build_graph(self, params, name=None):
        from .graph import DiTGraph
        return DiTGraph(params, name or "eraserdit")

    def build_workflow(self, params, **kwargs):
        from .workflow import build_workflow
        return build_workflow(params, **kwargs)

    def server_config(self, **kwargs):
        from .graph import server_config
        return server_config(**kwargs)

    def build_config(self, params):
        from nndeploy.e_dit.algorithms.eraserdit.config import EraserDiTPipelineConfig
        from nndeploy.e_dit.config.server_args import ServerArgs
        excluded = {'parallel_context', 'distributed_context', 'official_parallel_context'}
        if excluded.intersection(params.options):
            raise ValueError('Runtime communication handles cannot be supplied as Params')
        pipeline = EraserDiTPipelineConfig(**deepcopy(params.pipeline))
        options = dict(device='cuda', weight_dtype='bf16',
                       pipeline_class_name='EraserDiTErasePipeline',
                       component_architectures=dict(pipeline.component_architectures))
        options.update(deepcopy(params.options))
        options["algorithm"] = params.model
        return ServerArgs(model_path=params.model_path, pipeline_config=pipeline, **options)

    def validate_request(self, sampling, config):
        from nndeploy.e_dit.algorithms.eraserdit.validation import validate_request
        validate_request({'sampling': asdict(sampling)}, config)

    def config_scope(self, config):
        from nndeploy.e_dit.config.server_args import config_scope
        return config_scope(config)


adapter = EraserDiTAdapter()
