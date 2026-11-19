"""Bounded NTHWC normalization and NCTHW reduction primitives."""
import torch
from diffusers.models.normalization import RMSNorm


class ChunkedRMSNorm(RMSNorm):
    """Run the original channel reduction on bounded groups of video frames.

    VAE callers pass NTHWC views. No reduction crosses the temporal axis, so
    every chunk keeps the complete normalization dimension and original dtype
    conversions. The output retains the input layout for subsequent convolutions.
    Training and small tensors use the upstream implementation.
    """

    chunk_size = 0

    def forward(self, hidden_states):
        if (not self.chunk_size or torch.is_grad_enabled()
                or hidden_states.ndim != 5 or hidden_states.numel() <= self.chunk_size):
            return super().forward(hidden_states)
        frames = hidden_states.shape[1]
        per_frame = hidden_states.numel() // frames
        step = max(1, self.chunk_size // per_frame)
        output = None
        for start in range(0, frames, step):
            value = super().forward(hidden_states[:, start:start + step])
            if output is None:
                output = torch.empty_like(hidden_states, dtype=value.dtype)
            output[:, start:start + step].copy_(value)
            del value
        return output


def modulate(hidden, scale, shift, *, inplace=False):
    """Overwrite owned normalization output, preserving both native dtype rounds."""
    if inplace and not torch.is_grad_enabled() and hidden.dtype == scale.dtype == shift.dtype:
        return hidden.mul_(1 + scale).add_(shift)
    return hidden * (1 + scale) + shift


def apply_activation(hidden, activation, *, inplace=False):
    if inplace and not torch.is_grad_enabled() and isinstance(activation, torch.nn.SiLU):
        return torch.nn.functional.silu(hidden, inplace=True)
    return activation(hidden)


def chunked_downsample_residual(inputs, stride, group_size, chunk_elements):
    """Keep each channel/stride reduction complete, bound only output frames."""
    frames = inputs.shape[2] // stride[0]
    per_frame = inputs.numel() // frames
    step = max(1, chunk_elements // per_frame)
    output = None
    for start in range(0, frames, step):
        value = inputs[:, :, start * stride[0]:min(frames, start + step) * stride[0]]
        value = value.unflatten(4, (-1, stride[2])).unflatten(3, (-1, stride[1])).unflatten(2, (-1, stride[0]))
        value = value.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        value = value.unflatten(1, (-1, group_size)).mean(dim=2)
        if output is None:
            shape = list(value.shape)
            shape[2] = frames
            output = value.new_empty(shape)
        output[:, :, start:start + step].copy_(value)
    return output


def chunked_temporal_conv(conv, inputs, *, left, right, chunk_elements):
    """NCTHW Conv3d with explicit replicate temporal padding and intact receptive fields.

    The convolution must not apply temporal padding itself. The budget bounds
    input frames, with a full receptive field as the minimum allocation.
    """
    if chunk_elements < 1 or left < 0 or right < 0 or conv.padding[0] != 0:
        raise ValueError('positive chunk budget and external temporal padding required')
    kernel = conv.kernel_size[0]
    frames = inputs.shape[2]
    extent = conv.dilation[0] * (kernel - 1) + 1
    stride = conv.stride[0]
    output_frames = (frames + left + right - extent) // stride + 1
    per_frame = inputs.numel() // frames
    step = max(1, chunk_elements // per_frame)
    output = None
    for start in range(0, output_frames, step):
        end = min(start + step, output_frames)
        first = start * stride - left
        last = (end - 1) * stride - left + extent
        value = inputs[:, :, max(0, first):min(frames, last)]
        parts = []
        if first < 0:
            parts.append(inputs[:, :, :1].expand(-1, -1, -first, -1, -1))
        parts.append(value)
        if last > frames:
            parts.append(inputs[:, :, -1:].expand(-1, -1, last - frames, -1, -1))
        value = torch.cat(parts, dim=2) if len(parts) > 1 else value
        value = conv(value)
        if output is None:
            shape = list(value.shape)
            shape[2] = output_frames
            output = value.new_empty(shape)
        output[:, :, start:end].copy_(value)
        del value, parts
    return output

