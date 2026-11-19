"""Encode generated RGB frames and optional audio into a self-contained MP4."""
from fractions import Fraction
from io import BytesIO
import numpy as np
from .media import frame_rate, rgb_frames, audio_samples, VideoMetadata

def encode_audio_video(video, *, fps, audio=None, sample_rate=None):
    import av
    video = rgb_frames(video)
    metadata = VideoMetadata(video.shape[2], video.shape[1], len(video), frame_rate(fps))
    fps = metadata.fps
    samples = audio_samples(audio, sample_rate) if audio is not None else None
    buffer = BytesIO()
    with av.open(buffer, mode='w', format='mp4') as container:
        stream = container.add_stream('libx264', rate=fps)
        stream.width, stream.height = video.shape[2], video.shape[1]
        stream.pix_fmt = 'yuv420p'
        # Declare every stream before the muxer writes the container header.
        if samples is not None:
            layout = 'mono' if samples.shape[0] == 1 else 'stereo'
            track = container.add_stream('aac', rate=int(sample_rate))
            track.layout = layout
        for index, pixels in enumerate(video):
            frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            frame.pts, frame.time_base = index, 1 / fps
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
        if samples is not None:
            for offset in range(0, samples.shape[1], 1024):
                frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples[:,offset:offset+1024]),format='fltp',layout=layout)
                frame.sample_rate = int(sample_rate)
                frame.pts, frame.time_base = offset, Fraction(1, int(sample_rate))
                for packet in track.encode(frame): container.mux(packet)
            for packet in track.encode(): container.mux(packet)
    return buffer.getvalue()
