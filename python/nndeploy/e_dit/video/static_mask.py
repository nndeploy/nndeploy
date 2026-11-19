"""Expand a browser-painted PNG into a lossless, frame-aligned mask video."""
import json
from fractions import Fraction
from pathlib import Path
import subprocess

from PIL import Image



def expand_static_mask(video: Path, mask: Path, directory: Path) -> Path:
    with mask.open('rb') as stream:
        if stream.read(8) != b'\x89PNG\r\n\x1a\n':
            return mask
    output = directory / 'mask_static.mkv'
    try:
        probe = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
            '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames',
            '-of', 'json', str(video),
        ], check=True, capture_output=True, text=True, timeout=120)
        metadata = json.loads(probe.stdout)['streams'][0]
        frames = int(metadata['nb_read_frames'])
        rate = Fraction(metadata['avg_frame_rate'])
        if frames < 1 or rate <= 0:
            raise ValueError('video has no frames or valid frame rate')
        with Image.open(mask) as image:
            if image.format != 'PNG' or image.size != (metadata['width'], metadata['height']):
                raise ValueError('PNG mask dimensions must match the video')
            if getattr(image, 'n_frames', 1) != 1:
                raise ValueError('use a single PNG frame or a mask video')
            # Flatten alpha against black; transparent pixels must not erase.
            rgba = image.convert('RGBA')
            flat = Image.new('RGB', image.size, 'black')
            flat.paste(rgba, mask=rgba.getchannel('A'))
            png = directory / 'mask_static.png'
            flat.save(png)
        subprocess.run([
            'ffmpeg', '-v', 'error', '-nostdin', '-threads', '1',
            '-loop', '1', '-framerate', str(rate), '-i', str(png),
            '-frames:v', str(frames), '-an', '-c:v', 'ffv1', '-pix_fmt', 'gray',
            '-threads', '1', '-n', str(output),
        ], check=True, capture_output=True, timeout=120)
        return output
    except (ValueError, KeyError, IndexError, OSError, Image.DecompressionBombError,
            subprocess.SubprocessError) as error:
        output.unlink(missing_ok=True)
        raise ValueError('Cannot prepare PNG mask: ' + str(error)) from error
