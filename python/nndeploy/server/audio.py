"""Bounded audio samples for existing numeric fields and managed files."""
from fractions import Fraction
import struct

from .errors import ServerError

LAYOUTS = ('mono', 'stereo', '2.1', 'quad', '5.0', '5.1', '6.1', '7.1')
FORMATS = {'audio/wav': ('wav', 'pcm_f32le'), 'audio/flac': ('flac', 'flac'),
           'audio/mpeg': ('mp3', 'libmp3lame'), 'audio/ogg': ('ogg', 'libopus'),
           'audio/mp4': ('mp4', 'aac')}


def _av():
    try:
        import av
        return av
    except ImportError as error:
        raise ServerError(503, 'audio_unavailable', 'Audio sample conversion requires nndeploy[server-audio]') from error


def channels(field):
    return 1 if len(field.shape) == 1 else field.shape[1]


def check_config(config):
    if not any(field.is_audio() for group in (config.inputs, config.parameters, config.outputs) for field in group.values()):
        return
    av = _av()
    for field in config.outputs.values():
        if field.is_audio():
            try:
                codec = av.Codec(FORMATS[field.media_type][1], 'w')
            except (ValueError, av.FFmpegError) as error:
                raise ServerError(503, 'audio_unavailable', 'Configured audio encoder is unavailable') from error
            if codec.audio_rates and field.sample_rate not in codec.audio_rates:
                raise ValueError('Audio encoder does not support the configured sample_rate')
            if field.media_type == 'audio/mpeg' and channels(field) > 2:
                raise ValueError('MP3 output supports mono or stereo')


def _wave_size(stream, size):
    """A short PCM WAV must not silently become a shorter valid waveform."""
    header = stream.read(12)
    if len(header) != 12 or header[:4] != b'RIFF' or header[8:] != b'WAVE' or struct.unpack('<I', header[4:8])[0] + 8 != size:
        raise ValueError('Audio WAV requires a complete bounded RIFF container')
    data = False
    while stream.tell() < size:
        chunk = stream.read(8)
        if len(chunk) != 8:
            raise ValueError('Truncated WAV chunk')
        length = struct.unpack('<I', chunk[4:])[0]
        end = stream.tell() + length + length % 2
        if end > size:
            raise ValueError('Truncated WAV payload')
        if chunk[:4] == b'data':
            if not length or data:
                raise ValueError('WAV requires one nonempty sample payload')
            data = True
        stream.seek(end)
    if not data:
        raise ValueError('WAV sample payload is missing')
    stream.seek(0)


def read_audio(stream, field, *, load=True):
    import numpy as np
    from .arrays import check_shape, _check_numbers
    av = _av()
    stream.seek(0, 2)
    size = stream.tell()
    if not 0 < size <= 64 * 1024**2:
        raise ValueError('Encoded audio exceeds 64 MiB')
    stream.seek(0)
    if field.media_type == 'audio/wav':
        _wave_size(stream, size)
    count, parts = 0, []
    count_channels = channels(field)
    limit = min(field.max_elements // count_channels, field.max_bytes // (4 * count_channels), field.sample_rate * 600)
    if field.shape[0] is not None:
        limit = min(limit, field.shape[0])
    def consume(frame):
        nonlocal count
        if count + frame.samples > limit:
            raise ValueError('Decoded audio exceeds the configured sample or byte limit')
        array = frame.to_ndarray().reshape(-1, count_channels)
        _check_numbers(array, field)
        count += frame.samples
        if load:
            parts.append(array.copy())
    try:
        with av.open(stream, mode='r', options={'format_whitelist': 'wav,mp3,ogg,flac,mov',
                'protocol_whitelist': 'file,pipe', 'err_detect': 'explode', 'enable_drefs': '0'}) as container:
            expected = 'mov' if field.media_type == 'audio/mp4' else FORMATS[field.media_type][0]
            if expected not in container.format.name.split(',') or len(container.streams) != 1 or len(container.streams.audio) != 1:
                raise ValueError('Audio samples require one audio stream in the declared container')
            source = container.streams.audio[0]
            source.codec_context.thread_count = 1
            rate, source_channels = source.codec_context.sample_rate, source.codec_context.channels
            if not 0 < rate <= 192000 or not 1 <= source_channels <= 8:
                raise ValueError('Audio source sample rate or channels exceed supported limits')
            if container.duration is not None and not 0 < container.duration / av.time_base <= 600:
                raise ValueError('Audio duration must be at most 600 seconds')
            resampler = av.AudioResampler(format='flt', layout=LAYOUTS[count_channels - 1], rate=field.sample_rate)
            input_samples = 0
            for frame in container.decode(source):
                input_samples += frame.samples
                if frame.sample_rate != rate or len(frame.layout.channels) != source_channels or input_samples > rate * 600:
                    raise ValueError('Audio format changed or duration exceeds the limit')
                if frame.is_corrupt:
                    raise ValueError('Corrupt audio frame')
                # Reject NaN before a resampler or integer encoder can hide it.
                if frame.format.name.startswith(('flt', 'dbl')) and not np.isfinite(frame.to_ndarray()).all():
                    raise ValueError('Audio samples must be finite')
                frame.pts = None
                for converted in resampler.resample(frame):
                    consume(converted)
            for converted in resampler.resample(None):
                consume(converted)
    except av.FFmpegError as error:
        raise ValueError('Could not decode the declared audio file') from error
    if not count:
        raise ValueError('Audio requires nonempty samples')
    shape = [count] if len(field.shape) == 1 else [count, count_channels]
    check_shape(field, shape)
    return np.concatenate(parts).reshape(shape) if load else None


class _BoundedOutput:
    def __init__(self, stream, limit):
        self.stream, self.limit = stream, limit

    def write(self, value):
        if self.stream.tell() + len(value) > self.limit:
            raise ValueError('Encoded audio exceeds the file byte limit')
        return self.stream.write(value)

    def seek(self, offset, whence=0):
        result = self.stream.seek(offset, whence)
        if result < 0 or result > self.limit:
            raise ValueError('Audio output seek exceeds the file byte limit')
        return result

    def tell(self):
        return self.stream.tell()


def write_audio(stream, array, field, *, max_bytes=64 * 1024**2):
    import numpy as np
    from .arrays import check_shape, _check_numbers
    av = _av()
    check_shape(field, list(array.shape))
    _check_numbers(array, field)
    if array.dtype.name != 'float32' or not 0 < array.shape[0] <= field.sample_rate * 600:
        raise ValueError('Audio output requires nonempty bounded float32 samples')
    if field.media_type != 'audio/wav' and np.any(np.abs(array) > 1):
        raise ValueError('Compressed audio output samples must be between -1 and 1')
    format_name, codec_name = FORMATS[field.media_type]
    layout = LAYOUTS[channels(field) - 1]
    try:
        with av.open(_BoundedOutput(stream, max_bytes), 'w', format=format_name) as container:
            target = container.add_stream(codec_name, rate=field.sample_rate)
            target.layout = layout
            target.codec_context.thread_count = 1
            # Process frames in bounded blocks; the caller already owns an
            # independent snapshot before an async engine releases its buffer.
            for start in range(0, array.shape[0], 4096):
                block = np.ascontiguousarray(array[start:start + 4096].reshape(1, -1), dtype=np.float32)
                frame = av.AudioFrame.from_ndarray(block, format='flt', layout=layout)
                frame.sample_rate = field.sample_rate
                frame.time_base = Fraction(1, field.sample_rate)
                frame.pts = start
                for packet in target.encode(frame):
                    container.mux(packet)
            for packet in target.encode(None):
                container.mux(packet)
    except av.FFmpegError as error:
        raise ValueError('Could not encode the configured audio format') from error
