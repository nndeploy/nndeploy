"""Validate declared media before publishing a managed file."""
import json
import shutil
import subprocess
import warnings

from ..config import load_json
from ..errors import ServerError


MEDIA_EXTENSIONS = {
    'image/png': '.png', 'image/jpeg': '.jpg', 'image/webp': '.webp',
    'text/plain': '.txt', 'application/json': '.json', 'application/octet-stream': '.bin',
    'application/x-npy': '.npy',
    'application/zip': '.zip',
    'audio/wav': '.wav', 'audio/mpeg': '.mp3', 'audio/ogg': '.ogg', 'audio/flac': '.flac',
    'audio/mp4': '.m4a', 'video/mp4': '.mp4', 'video/webm': '.webm',
}


def validate(path, media_type=None, *, max_archive_bytes=None):
    from PIL import Image
    media_type = (media_type or 'image/*').split(';', 1)[0].strip().lower()
    if media_type == 'application/zip':
        import zipfile
        import zlib
        from .archive import validate_archive
        try:
            validate_archive(path, max_bytes=max_archive_bytes)
        except (OSError, ValueError, EOFError, RuntimeError, zipfile.BadZipFile, zlib.error) as exc:
            raise ServerError(422, 'invalid_archive', 'File is not a supported bounded ZIP archive') from exc
        return '.zip', media_type
    if media_type == 'application/x-npy':
        from ..arrays import read_npy
        try:
            with path.open('rb') as stream:
                read_npy(stream, load=False)
        except (OSError, ValueError, TypeError, OverflowError, EOFError) as exc:
            raise ServerError(422, 'invalid_array', 'File is not a supported, bounded numeric NPY array') from exc
        return '.npy', media_type
    if media_type.startswith('image/'):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(path) as image:
                    actual = {'PNG': ('.png', 'image/png'), 'JPEG': ('.jpg', 'image/jpeg'), 'WEBP': ('.webp', 'image/webp')}.get(image.format)
                    if actual is None or (media_type != 'image/*' and media_type != actual[1]):
                        raise ValueError('Image type does not match content')
                    if image.width * image.height > 32_000_000:
                        raise ValueError('Image pixel limit exceeded')
                    image.verify()
            return actual
        except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ServerError(422, 'invalid_image', 'File is not a supported, bounded image') from exc
    if media_type in ('text/plain', 'application/json'):
        try:
            if path.stat().st_size > 4 * 1024**2:
                raise ValueError('Text and JSON are limited to 4 MiB')
            text = path.read_text(encoding='utf-8')
            if '\x00' in text:
                raise ValueError('Text cannot contain NUL characters')
            if media_type == 'application/json':
                load_json(path)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise ServerError(422, 'invalid_text', 'File is not valid bounded UTF-8 text or JSON') from exc
        return ('.json' if media_type == 'application/json' else '.txt', media_type)
    if media_type == 'application/octet-stream':
        return '.bin', media_type
    supported = {'audio/wav': '.wav', 'audio/mpeg': '.mp3', 'audio/ogg': '.ogg', 'audio/flac': '.flac',
                 'audio/mp4': '.m4a', 'video/mp4': '.mp4', 'video/webm': '.webm'}
    if media_type not in supported:
        raise ServerError(415, 'unsupported_media', 'Unsupported declared media type')
    probe = shutil.which('ffprobe')
    if probe is None:
        raise ServerError(503, 'media_unavailable', 'Audio and video validation requires ffprobe')
    try:
        result = subprocess.run([probe, '-v', 'error', '-protocol_whitelist', 'file,pipe',
            '-format_whitelist', 'mov,matroska,webm,wav,mp3,ogg,flac', '-show_format', '-show_streams', '-of', 'json', str(path)],
            capture_output=True, timeout=10, check=True)
        info = json.loads(result.stdout)
        duration = float(info['format'].get('duration', 0))
        if not 0 < duration <= 600 or len(info['streams']) > 8:
            raise ValueError('Invalid duration or stream count')
        kinds = {stream.get('codec_type') for stream in info['streams']}
        if media_type.startswith('video/') and 'video' not in kinds:
            raise ValueError('Video requires a video stream')
        if media_type.startswith('audio/') and ('audio' not in kinds or 'video' in kinds):
            raise ValueError('Audio requires audio streams only')
        formats = set(info['format']['format_name'].split(','))
        required = {'audio/wav': 'wav', 'audio/mpeg': 'mp3', 'audio/ogg': 'ogg', 'audio/flac': 'flac',
                    'audio/mp4': 'mov', 'video/mp4': 'mov', 'video/webm': 'webm'}[media_type]
        if required not in formats:
            raise ValueError('Media format does not match declared type')
        for stream in info['streams']:
            if stream.get('width', 0) > 4096 or stream.get('height', 0) > 4096 or stream.get('channels', 0) > 8 or int(stream.get('sample_rate', 0)) > 192000:
                raise ValueError('Media dimensions or channel limits exceeded')
    except (subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        raise ServerError(422, 'invalid_media', 'Invalid or unsupported bounded audio/video') from exc
    return supported[media_type], media_type
