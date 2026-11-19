"""Bounded, cancellable input materialization for algorithm services."""
from pathlib import Path
from urllib.parse import urlparse, unquote
import requests


def materialize(source, target, *, check=lambda: None, max_bytes=5 * 1024**3, object_client=None,
                default_bucket=None, endpoint=None):
    check()
    target = Path(target)
    parsed = urlparse(source)
    if not parsed.scheme and Path(source).is_file():
        if Path(source).stat().st_size > max_bytes:
            raise ValueError('Input source exceeds the file size limit')
        return str(Path(source).absolute())
    response = None
    if parsed.scheme == 's3' or (object_client is not None and parsed.scheme not in ('http', 'https')):
        if object_client is None:
            raise ValueError('Object storage credentials are required for s3 inputs')
        bucket = parsed.netloc or default_bucket
        key = parsed.path.lstrip('/') if parsed.scheme else source.lstrip('/')
        stream = object_client.get_object(Bucket=bucket, Key=unquote(key))['Body']
    elif object_client is not None and parsed.netloc == urlparse(endpoint or '').netloc:
        parts = parsed.path.lstrip('/').split('/', 1)
        key = parts[1] if len(parts) == 2 and parts[0] == default_bucket else parsed.path.lstrip('/')
        stream = object_client.get_object(Bucket=default_bucket, Key=unquote(key))['Body']
    elif parsed.scheme in ('http', 'https'):
        response = requests.get(source, stream=True, timeout=(10, 30))
        try:
            response.raise_for_status()
        except requests.HTTPError:
            status = response.status_code
            response.close()
            raise ValueError(f'Input download returned HTTP {status}') from None
        stream = response.raw
        stream.decode_content = True
    else:
        raise FileNotFoundError('Input source does not exist')
    try:
        count = 0
        with target.open('wb') as output:
            while True:
                check()
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > max_bytes:
                    raise ValueError('Input source exceeds the file size limit')
                output.write(chunk)
        check()
        return str(target)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    finally:
        stream.close()
        if response is not None:
            response.close()
