"""Per-frame probe regions; preserve small mask and boundary changes in cache metrics."""
import torch
import torch.nn.functional as F


class ProbeRegions:
    def __init__(self, mask, *, shape, shard=slice(None)):
        frames, height, width = shape
        if mask is None or tuple(mask.shape[1:]) != (1, frames, height, width):
            raise ValueError('mask_frame_max requires a single-channel latent mask matching the token grid')
        binary = mask[:, 0] > 0
        planes = binary.reshape(-1, 1, height, width).float()
        dilated = F.max_pool2d(planes, 3, stride=1, padding=1) > 0
        eroded = 1 - F.max_pool2d(1 - planes, 3, stride=1, padding=1)
        boundary = (dilated & (eroded == 0)).reshape_as(binary)
        self.weights = torch.stack((torch.ones_like(binary), binary, boundary), dim=-1).flatten(1, 3)[:, shard]
        self.frame_ids = torch.arange(frames, device=mask.device).repeat_interleave(height * width)[shard]
        self.frames = frames

    def sums(self, current, previous):
        if current.shape != previous.shape or current.shape[:2] != self.weights.shape[:2]:
            raise ValueError('probe region and activation layouts must match')
        # Reduce channels before materializing region weights: no [B,N,C,R] temporary.
        previous = previous.float()
        numerator = (current.float() - previous).abs().sum(-1)
        denominator = previous.abs().sum(-1)
        values = torch.stack((numerator, denominator), dim=-1)
        weighted = values[:, :, None, :] * self.weights[:, :, :, None]
        sums = torch.zeros(current.shape[0], self.frames, 3, 2, device=current.device, dtype=torch.float64)
        sums.index_add_(1, self.frame_ids, weighted.double())
        return sums

    @staticmethod
    def distance(sums):
        # Empty regions contribute 0. Non-finite probes remain non-finite and
        # force compute in the existing controllers.
        return (sums[..., 0] / sums[..., 1].clamp_min(1e-12)).max().item()
