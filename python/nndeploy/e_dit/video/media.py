"""Explicit media boundary contracts; no decoder or model dependencies."""
from dataclasses import dataclass
from fractions import Fraction
import math
import numpy as np


def frame_rate(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)) or not math.isfinite(value) or value <= 0:
        raise ValueError('fps must be positive and finite')
    return Fraction(str(value)).limit_denominator(1000000)


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    num_frames: int
    fps: Fraction
    layout: str = 'FHWC'
    color_space: str = 'RGB'

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.width, self.height, self.num_frames)):
            raise ValueError('Video dimensions and frame count must be positive integers')
        object.__setattr__(self, 'fps', frame_rate(self.fps))
        if self.layout != 'FHWC' or self.color_space != 'RGB':
            raise ValueError('Encoded video boundary requires FHWC RGB')

    def to_dict(self):
        return dict(width=self.width, height=self.height, num_frames=self.num_frames,
                    fps_numerator=self.fps.numerator, fps_denominator=self.fps.denominator,
                    layout=self.layout, color_space=self.color_space)


def rgb_frames(video):
    """FHWC uint8 or finite float [0,1]; float values are clipped for encoding."""
    video = np.asarray(video)
    if video.ndim != 4 or video.shape[-1] != 3 or any(v <= 0 for v in video.shape):
        raise ValueError('Expected nonempty FHWC RGB video')
    if video.dtype == np.uint8:
        return video
    if not np.issubdtype(video.dtype, np.floating) or not np.isfinite(video).all():
        raise ValueError('RGB video must be uint8 or finite floating-point values')
    return (np.clip(video, 0, 1) * 255).round().astype(np.uint8)


def audio_samples(audio, sample_rate):
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError('Audio requires a positive integer sample rate')
    if hasattr(audio, 'detach'):
        audio = audio.detach().cpu().float().numpy()
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[None]
    if samples.ndim != 2 or samples.shape[0] not in (1, 2) or not samples.shape[1]:
        raise ValueError('Expected nonempty mono or stereo audio shaped CN')
    if not np.isfinite(samples).all():
        raise ValueError('Audio samples must be finite')
    return samples
