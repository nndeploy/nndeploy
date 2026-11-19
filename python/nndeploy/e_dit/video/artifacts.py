"""File output naming and input protection shared by video workflows."""
from pathlib import Path


def video_output_paths(path, *, include_crop=False):
    path = Path(path)
    if path.suffix.lower() not in ('.mp4', '.mov', '.m4v', '.mkv', '.webm', '.avi'):
        raise ValueError('Video output requires a supported video extension')
    paths = [str(path)]
    if include_crop:
        paths.append(str(path.with_name(path.stem + '_crop_only' + path.suffix)))
    return paths


def validate_output_paths(outputs, inputs):
    """Reject overwriting an input, including symlink and hardlink aliases."""
    sources = [Path(path).resolve() for path in inputs if path]
    targets = [Path(path).resolve() for path in outputs]
    if len(set(targets)) != len(targets):
        raise ValueError('Video output paths must be distinct')
    for target in targets:
        for source in sources:
            if target == source or (target.exists() and source.exists() and target.samefile(source)):
                raise ValueError(f'Video output would overwrite an input: {target}')
