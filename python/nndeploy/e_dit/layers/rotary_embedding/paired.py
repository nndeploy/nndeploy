"""RoPE with separate half-width cosine/sine tables and explicit pairing."""
import torch

def apply_rotary_pair(x, cos, sin, is_neox_style=False, interleaved=False):
    # Tables contain one angle per pair; GPT-J pairs adjacent channels.
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    a, b = x.chunk(2, -1) if is_neox_style else (x[..., ::2], x[..., 1::2])
    real = (a.float()*cos-b.float()*sin).to(x.dtype)
    imag = (b.float()*cos+a.float()*sin).to(x.dtype)
    return torch.cat((real,imag),-1) if is_neox_style else torch.stack((real,imag),-1).flatten(-2)

def apply_rotary_qk(query, key, cos_sin_cache, *, is_neox=False):
    cos,sin=cos_sin_cache.chunk(2,-1)
    return apply_rotary_pair(query,cos,sin,is_neox),apply_rotary_pair(key,cos,sin,is_neox)
