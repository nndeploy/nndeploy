"""Bounded-memory optional postprocessing of native video file artifacts."""
from pathlib import Path


def process_files(paths, request, context):
    if not (request.enable_frame_interpolation or request.enable_upscaling):
        return paths
    from ..video.stream_io import probe_video, SequentialReader, encode_profile, VideoWriterFfmpeg
    from ..video.ffmpeg_io import mux_source_audio
    interpolator = upscaler = None
    if request.enable_frame_interpolation:
        from nndeploy.e_dit.video.postprocess.rife_interpolator import FrameInterpolator
        interpolator = FrameInterpolator(request.frame_interpolation_model_path)
    if request.enable_upscaling:
        from nndeploy.e_dit.video.postprocess.realesrgan_upscaler import ImageUpscaler
        upscaler = ImageUpscaler(request.upscaling_model_path, scale=request.upscaling_scale)
    for index, path in enumerate(paths):
        context.check()
        meta = probe_video(str(path))
        multiplier = 2**request.frame_interpolation_exp if interpolator and meta['num_frame'] > 1 else 1
        temporary = context.directory / f'postprocess-{index}{Path(path).suffix}'
        writer = None
        try:
            with SequentialReader(meta) as reader:
                previous = reader.read_range(0, 1)[0]
                for frame_index in range(1, meta['num_frame'] + 1):
                    context.check()
                    if frame_index < meta['num_frame']:
                        following = reader.read_range(frame_index, frame_index + 1)[0]
                        frames = (interpolator.interpolate([previous, following],
                            exp=request.frame_interpolation_exp, scale=request.frame_interpolation_scale)[0][:-1]
                            if interpolator else [previous])
                        previous = following
                    else:
                        frames = [previous]
                    for frame in frames:
                        context.check()
                        if upscaler:
                            frame = upscaler.upscale([frame])[0]
                        if writer is None:
                            # Like the source VideoEdit writer, preserve the reference
                            # codec, bitrate and colour metadata over generic quality.
                            writer = VideoWriterFfmpeg(str(temporary), frame.shape[1], frame.shape[0],
                                meta['fps'] * multiplier, profile=encode_profile(meta))
                        writer.write(frame)
                    context.publish({'progress': 98, 'stage': 'postprocessing',
                                     'frame': frame_index, 'total_frames': meta['num_frame']})
            writer.close()
            count = (meta['num_frame'] - 1) * multiplier + 1
            if request.preserve_audio:
                mux_source_audio(str(temporary), str(path), count / (meta['fps'] * multiplier))
            context.check()
            # copyfile also works when the requested output is on another mount.
            import shutil
            shutil.copyfile(temporary, path)
        finally:
            if writer is not None:
                writer.close()
            temporary.unlink(missing_ok=True)
    return paths
