from diffusers.utils import is_torch_available


if is_torch_available():
    from nndeploy.e_dit.models.hooks.group_offloading import apply_group_offloading
    from nndeploy.e_dit.models.hooks.hooks import HookRegistry, ModelHook
    from nndeploy.e_dit.models.hooks.layerwise_casting import apply_layerwise_casting, apply_layerwise_casting_hook
    from nndeploy.e_dit.models.hooks.pyramid_attention_broadcast import PyramidAttentionBroadcastConfig, apply_pyramid_attention_broadcast
