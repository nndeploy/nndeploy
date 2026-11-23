"""Provider SDK operations behind Server file storage; credentials stay in the environment."""
import os
from contextlib import contextmanager
from datetime import timedelta
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator
from ..config import StrictModel
from ..errors import ServerError


class StorageConfig(StrictModel):
    backend: Literal['local', 's3', 'cos', 'oss', 'obs'] = 'local'
    bucket: str | None = None
    region: str | None = None
    endpoint: str | None = None
    prefix: str = 'nndeploy'
    access_key_env: str | None = None
    secret_key_env: str | None = None
    token_env: str | None = None
    max_bytes: int = Field(default=16 * 1024 * 1024, ge=1, le=5 * 1024**3)
    media_max_duration_seconds: float = Field(default=600, gt=0, le=86400, allow_inf_nan=False)
    media_max_dimension: int = Field(default=4096, ge=1, le=32768)
    retention_seconds: float | None = Field(default=None, ge=60, allow_inf_nan=False)
    max_files_per_subject: int | None = Field(default=None, ge=1, le=1_000_000)
    max_storage_bytes_per_subject: int | None = Field(default=None, ge=1, le=2**63 - 1)

    @model_validator(mode='after')
    def check(self):
        if self.backend != 'local' and not self.bucket:
            raise ValueError('Object storage requires a bucket')
        if self.backend in ('cos', 'oss') and not self.region:
            raise ValueError('COS and OSS require a region')
        if self.backend == 'obs' and not self.endpoint:
            raise ValueError('OBS requires an endpoint')
        if self.bucket and any(c in self.bucket for c in ('/', '\\', '\x00')):
            raise ValueError('Invalid bucket name')
        if any(part in ('', '.', '..') for part in self.prefix.split('/')) or '\\' in self.prefix:
            raise ValueError('Storage prefix must be a relative object prefix')
        if bool(self.access_key_env) != bool(self.secret_key_env):
            raise ValueError('Configure both credential environment variable names')
        if self.endpoint:
            url = urlsplit(self.endpoint)
            if url.username or url.password or url.query or url.fragment:
                raise ValueError('Storage endpoint cannot contain credentials or query parameters')
            if not url.hostname or (url.scheme != 'https' and not (url.scheme == 'http' and url.hostname in ('localhost', '127.0.0.1', '::1'))):
                raise ValueError('Storage endpoint must use HTTPS, except local test endpoints')
        return self


def _env(name):
    if name is None:
        return None
    value = os.environ.get(name)
    if not value:
        raise ValueError(f'Missing credential environment variable: {name}')
    return value


class ObjectBackend:
    def __init__(self, config):
        self.config = config
        provider = config.backend
        access, secret, token = _env(config.access_key_env), _env(config.secret_key_env), _env(config.token_env)
        if provider == 's3':
            import boto3
            from botocore.config import Config
            options = dict(region_name=config.region, endpoint_url=config.endpoint,
                config=Config(connect_timeout=10, read_timeout=30, retries={'max_attempts': 3, 'mode': 'standard'},
                              s3={'addressing_style': 'path'} if config.endpoint else {}))
            if access:
                options.update(aws_access_key_id=access, aws_secret_access_key=secret, aws_session_token=token)
            self.client = boto3.client('s3', **options)
        elif provider == 'cos':
            from qcloud_cos import CosConfig, CosS3Client
            self.client = CosS3Client(CosConfig(Region=config.region, SecretId=access or _env('COS_SECRET_ID'),
                SecretKey=secret or _env('COS_SECRET_KEY'), Token=token, Scheme='https', Timeout=30,
                Endpoint=urlsplit(config.endpoint).netloc if config.endpoint else None, AllowRedirects=False))
        elif provider == 'oss':
            import alibabacloud_oss_v2 as oss
            credentials = (oss.credentials.StaticCredentialsProvider(access, secret, token) if access else
                           oss.credentials.EnvironmentVariableCredentialsProvider())
            self.client = oss.Client(oss.Config(region=config.region, endpoint=config.endpoint,
                credentials_provider=credentials, connect_timeout=10, readwrite_timeout=30, retry_max_attempts=3,
                enabled_redirect=False))
        elif provider == 'obs':
            from obs import ObsClient
            self.client = ObsClient(access_key_id=access or _env('OBS_ACCESS_KEY_ID'),
                secret_access_key=secret or _env('OBS_SECRET_ACCESS_KEY'), security_token=token,
                server=config.endpoint, timeout=30, max_retry_count=3)
        else:
            raise ValueError('ObjectBackend requires a remote provider')

    def key(self, value):
        return f'{self.config.prefix}/{value}'

    @staticmethod
    def _obs(response):
        if response.status >= 300:
            raise ServerError(503, 'storage_unavailable', 'OBS operation failed', retryable=True)
        return response

    def put_file(self, key, path, media_type):
        bucket, key, provider = self.config.bucket, self.key(key), self.config.backend
        if provider == 's3':
            from boto3.s3.transfer import TransferConfig
            self.client.upload_file(str(path), bucket, key, ExtraArgs={'ContentType': media_type},
                Config=TransferConfig(multipart_threshold=8 * 1024**2, multipart_chunksize=8 * 1024**2, max_concurrency=2))
        elif provider == 'cos':
            self._cos_put_file(bucket, key, path, media_type)
        elif provider == 'oss':
            import alibabacloud_oss_v2 as oss
            self.client.put_object_from_file(oss.PutObjectRequest(bucket=bucket, key=key, content_type=media_type), str(path))
        else:
            from obs import PutObjectHeader
            self._obs(self.client.putFile(bucket, key, str(path), headers=PutObjectHeader(contentType=media_type)))

    def _cos_put_file(self, bucket, key, path, media_type):
        # COS upload_file deliberately leaves failed uploads resumable. Managed
        # files have unique keys and discard failed writes, so own the upload
        # explicitly instead of leaking incomplete parts on that SDK path.
        from concurrent.futures import ThreadPoolExecutor
        import logging
        part_size = 8 * 1024**2
        size = os.path.getsize(path)
        if size <= part_size:
            with open(path, 'rb') as source:
                self.client.put_object(Bucket=bucket, Key=key, Body=source,
                                       EnableMD5=True, ContentType=media_type)
            return
        upload_id = self.client.create_multipart_upload(
            Bucket=bucket, Key=key, ContentType=media_type)['UploadId']
        try:
            def upload(part):
                number, offset = part
                # Bound memory to two chunks, independent of the file size.
                with open(path, 'rb') as source:
                    source.seek(offset)
                    chunk = source.read(min(part_size, size - offset))
                result = self.client.upload_part(Bucket=bucket, Key=key, Body=chunk,
                    PartNumber=number, UploadId=upload_id, EnableMD5=True)
                return {'PartNumber': number, 'ETag': result['ETag']}
            # Join active writers before aborting; map cancels pending futures
            # if a part fails. Completion receives parts in numeric order.
            with ThreadPoolExecutor(max_workers=2) as pool:
                parts = list(pool.map(upload, enumerate(range(0, size, part_size), 1)))
            self.client.complete_multipart_upload(Bucket=bucket, Key=key,
                UploadId=upload_id, MultipartUpload={'Part': parts})
        except BaseException:
            try:
                self.client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
            except Exception:
                # SDK exception text may contain signed URLs. Keep diagnostics
                # credential-free and preserve the original upload failure.
                logging.getLogger(__name__).error('COS multipart abort failed for key %s, upload %s', key, upload_id)
            raise

    @contextmanager
    def open(self, key):
        bucket, key, provider = self.config.bucket, self.key(key), self.config.backend
        if provider == 's3':
            stream = self.client.get_object(Bucket=bucket, Key=key)['Body']
        elif provider == 'cos':
            stream = self.client.get_object(Bucket=bucket, Key=key)['Body'].get_raw_stream()
        elif provider == 'oss':
            import alibabacloud_oss_v2 as oss
            stream = self.client.get_object(oss.GetObjectRequest(bucket=bucket, key=key)).body
        else:
            stream = self._obs(self.client.getObject(bucket, key, loadStreamInMemory=False)).body.response
        try:
            yield stream
        finally:
            stream.close()

    def delete(self, key):
        bucket, key, provider = self.config.bucket, self.key(key), self.config.backend
        if provider == 's3':
            self.client.delete_object(Bucket=bucket, Key=key)
        elif provider == 'cos':
            self.client.delete_object(Bucket=bucket, Key=key)
        elif provider == 'oss':
            import alibabacloud_oss_v2 as oss
            self.client.delete_object(oss.DeleteObjectRequest(bucket=bucket, key=key))
        else:
            self._obs(self.client.deleteObject(bucket, key))

    def signed_get(self, key, expires):
        if not 1 <= expires <= 3600:
            raise ValueError('Download URL expiry must be between 1 and 3600 seconds')
        bucket, key, provider = self.config.bucket, self.key(key), self.config.backend
        if provider == 's3':
            return self.client.generate_presigned_url('get_object', Params={'Bucket': bucket, 'Key': key}, ExpiresIn=expires)
        if provider == 'cos':
            return self.client.get_presigned_url(Bucket=bucket, Key=key, Method='GET', Expired=expires)
        if provider == 'oss':
            import alibabacloud_oss_v2 as oss
            return self.client.presign(oss.GetObjectRequest(bucket=bucket, key=key), expires=timedelta(seconds=expires)).url
        return self.client.createSignedUrl('GET', bucket, key, expires=expires).signedUrl

    def close(self):
        close = getattr(self.client, 'close', None)
        if close:
            close()
