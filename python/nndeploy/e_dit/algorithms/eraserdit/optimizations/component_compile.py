"""EraserDiT component selection and T5/VAE compile precision policy."""
from nndeploy.e_dit.layers.component_compile import CompileTarget, ComponentCompileManager as CompileManager


class ComponentCompileManager(CompileManager):
    def __init__(self, modules, names, *, mode='max-autotune-no-cudagraphs'):
        targets = []
        for name in names:
            if name == 'text_encoder':
                module = modules[name]
                if any(hasattr(child, '_get_fsdp_state') for child in module.modules()):
                    raise ValueError('cannot compile a text encoder with FSDP hooks')
            elif name in ('vae_encoder', 'vae_decoder'):
                module = getattr(modules['vae'], name.removeprefix('vae_'))
            else:
                raise ValueError(f'unsupported compile component: {name}')
            targets.append(CompileTarget(name, module, name != 'text_encoder'))
        super().__init__(targets, mode=mode)
