"""Reusable encoding operations; algorithms retain latent scaling and packing."""
import torch
from nndeploy.e_dit.memory.component_use import use_component


def encode_image(model, processor, image, *, device, offload=False, **forward_kwargs):
    if processor is None:
        raise ValueError('Image encoding requires an image processor')
    with use_component(model, device=device, offload=offload), torch.no_grad():
        parameter = next(model.parameters(), None)
        dtype = parameter.dtype if parameter is not None else None
        inputs = processor(images=image, return_tensors='pt')
        inputs = {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
                  for key, value in inputs.items()}
        return model(**inputs, **forward_kwargs)


def encode_vae(model, video, *, device, offload=False):
    """Return the deterministic posterior mode, with no algorithm normalization."""
    with use_component(model, device=device, offload=offload), torch.no_grad():
        parameter = next(model.parameters(), None)
        dtype = parameter.dtype if parameter is not None else video.dtype
        encoded = model.encode(video.to(device=device, dtype=dtype))
        posterior = getattr(encoded, 'latent_dist', encoded)
        if not callable(getattr(posterior, 'mode', None)):
            raise TypeError('VAE encoder must return a posterior with mode()')
        return posterior.mode()
