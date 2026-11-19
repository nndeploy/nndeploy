# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang VSR/OpenAI video interfaces; see ../LICENSE.SGLang.
import asyncio
import os
import shutil
from typing import Any, Optional
from urllib.parse import unquote, urlparse

from logging import getLogger as init_logger

logger = init_logger(__name__)


_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".m4v": "video/x-m4v",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".glb": "model/gltf-binary",
    ".obj": "text/plain",
}


def normalize_endpoint(endpoint: str, secure: bool = False) -> str:
    endpoint = endpoint.strip().rstrip("/")
    if not endpoint:
        raise ValueError("S3 endpoint must not be empty")
    if endpoint.startswith(("http://", "https://")):
        return endpoint
    scheme = "https" if secure else "http"
    return f"{scheme}://{endpoint}"


def normalize_object_key(key: str) -> str:
    normalized = key.strip().lstrip("/")
    if not normalized:
        raise ValueError("S3 object key must not be empty")
    if any(part == ".." for part in normalized.split("/")):
        raise ValueError("S3 object key must not contain '..'")
    return normalized


def is_cos_storage_config(config: Any) -> bool:
    provider = str(getattr(config, "provider", None) or "").strip().lower()
    if provider:
        return provider == "cos"

    endpoint = normalize_endpoint(
        config.endpoint, bool(getattr(config, "secure", False))
    )
    hostname = (urlparse(endpoint).hostname or "").lower()
    return hostname.endswith(".myqcloud.com") and (
        hostname.startswith("cos.") or ".cos." in hostname
    )


def apply_object_key_prefix(key: str, prefix: Optional[str]) -> str:
    normalized_key = normalize_object_key(key)
    normalized_prefix = (prefix or "").strip().strip("/")
    if not normalized_prefix:
        return normalized_key

    normalized_prefix = normalize_object_key(normalized_prefix)
    if normalized_key == normalized_prefix or normalized_key.startswith(
        f"{normalized_prefix}/"
    ):
        return normalized_key
    return f"{normalized_prefix}/{normalized_key}"


def _content_type_for_path(local_path: str) -> str:
    ext = os.path.splitext(local_path)[1].lower()
    return _CONTENT_TYPES.get(ext, "application/octet-stream")


def _append_source_extension(
    target_path: str, source: str, default_ext: str = ".mp4"
) -> str:
    if os.path.splitext(target_path)[1]:
        return target_path
    parsed = urlparse(source)
    suffix_source = parsed.path if parsed.scheme else source
    _, ext = os.path.splitext(suffix_source)
    return f"{target_path}{ext or default_ext}"


def _url_netloc_without_auth(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.split("@")[-1].lower()


class CloudStorage:
    def __init__(self):
        self.enabled = os.getenv("SGLANG_CLOUD_STORAGE_TYPE", "").lower() == "s3"
        if not self.enabled:
            return

        try:
            import boto3
        except ImportError:
            logger.error(
                "boto3 is not installed. Please install it with `pip install boto3` to use cloud storage."
            )
            self.enabled = False
            return

        self.bucket_name = os.getenv("SGLANG_S3_BUCKET_NAME")
        if not self.bucket_name:
            self.enabled = False
            return

        endpoint_url = os.getenv("SGLANG_S3_ENDPOINT_URL") or None
        region_name = os.getenv("SGLANG_S3_REGION_NAME") or None

        self.client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("SGLANG_S3_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("SGLANG_S3_SECRET_ACCESS_KEY"),
            endpoint_url=endpoint_url,
            region_name=region_name,
        )
        self.endpoint_url = endpoint_url
        self.region_name = region_name

    def is_enabled(self) -> bool:
        return self.enabled

    async def upload_file(
        self,
        local_path: str,
        destination_key: str,
        bucket_name: Optional[str] = None,
    ) -> Optional[str]:
        if not self.is_enabled():
            return None

        destination_key = normalize_object_key(destination_key)
        target_bucket = bucket_name or self.bucket_name
        if not target_bucket:
            logger.error("Upload failed: S3 bucket is not configured")
            return None

        def _sync_upload():
            """Synchronous part of the upload to run in a thread."""
            self.client.upload_file(
                local_path,
                target_bucket,
                destination_key,
                ExtraArgs={"ContentType": _content_type_for_path(local_path)},
            )

        try:
            # Offload the blocking I/O call to a thread executor
            await asyncio.get_running_loop().run_in_executor(None, _sync_upload)
        except Exception as e:
            # If upload fails, log the error and return None for fallback
            logger.error(f"Upload failed for {destination_key}: {e}")
            return None

        # Simplified URL generation with a default region
        if self.endpoint_url:
            url = f"{self.endpoint_url.rstrip('/')}/{target_bucket}/{destination_key}"
        else:
            region = self.region_name or "us-east-1"
            url = f"https://{target_bucket}.s3.{region}.amazonaws.com/{destination_key}"

        logger.info(f"Uploaded {local_path} to {url}")
        return url

    async def upload_and_cleanup(
        self,
        file_path: str,
        destination_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
    ) -> Optional[str]:
        """Helper to upload a file and delete the local copy if successful."""
        if not self.is_enabled():
            return None

        key = destination_key or os.path.basename(file_path)
        url = await self.upload_file(file_path, key, bucket_name=bucket_name)

        if url:
            try:
                # pass if removal fails
                os.remove(file_path)
            except OSError as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {e}")
        return url


class RequestCloudStorage:
    """Per-request S3-compatible storage client for VideoEdit business requests."""

    def __init__(self, config: Any):
        try:
            import boto3
        except ImportError as e:
            raise RuntimeError(
                "boto3 is not installed. Please install it to use minioConfig."
            ) from e

        self.bucket_name = config.bucket_name
        self.endpoint_url = normalize_endpoint(config.endpoint, config.secure)
        self.region_name = config.region or "us-east-1"
        self.client = boto3.client(
            "s3",
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            endpoint_url=self.endpoint_url,
            region_name=self.region_name,
        )

    def _public_url(self, bucket_name: str, object_key: str) -> str:
        return f"{self.endpoint_url.rstrip('/')}/{bucket_name}/{object_key}"

    def _parse_source_object(self, source: str) -> tuple[str, str] | None:
        if source.startswith("s3://"):
            parsed = urlparse(source)
            bucket = parsed.netloc or self.bucket_name
            key = normalize_object_key(unquote(parsed.path))
            return bucket, key

        parsed = urlparse(source)
        if parsed.scheme not in {"http", "https"}:
            return self.bucket_name, normalize_object_key(source)

        configured_netloc = _url_netloc_without_auth(self.endpoint_url)
        source_netloc = _url_netloc_without_auth(source)
        if source_netloc != configured_netloc:
            return None

        parts = [unquote(part) for part in parsed.path.split("/") if part]
        if not parts:
            return None
        if parts[0] == self.bucket_name:
            key_parts = parts[1:]
            if not key_parts:
                return None
            return self.bucket_name, normalize_object_key("/".join(key_parts))
        return self.bucket_name, normalize_object_key("/".join(parts))

    async def download_source(
        self,
        source: str,
        target_path: str,
        *,
        default_ext: str = ".mp4",
        timeout: float = 60.0,
    ) -> str:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        target_path = _append_source_extension(target_path, source, default_ext)

        if os.path.exists(source):
            if os.path.abspath(source) == os.path.abspath(target_path):
                return source
            shutil.copyfile(source, target_path)
            return target_path

        object_ref = self._parse_source_object(source)
        if object_ref is not None:
            bucket_name, object_key = object_ref

            def _sync_download():
                self.client.download_file(bucket_name, object_key, target_path)

            await asyncio.get_running_loop().run_in_executor(None, _sync_download)
            return target_path

        if source.lower().startswith(("http://", "https://")):
            import httpx

            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(source, timeout=timeout)
                response.raise_for_status()
            with open(target_path, "wb") as f:
                f.write(response.content)
            return target_path

        raise FileNotFoundError(f"Input source does not exist: {source}")

    async def upload_file(
        self,
        local_path: str,
        destination_key: str,
        bucket_name: Optional[str] = None,
    ) -> str:
        destination_key = normalize_object_key(destination_key)
        target_bucket = bucket_name or self.bucket_name

        def _sync_upload():
            self.client.upload_file(
                local_path,
                target_bucket,
                destination_key,
                ExtraArgs={"ContentType": _content_type_for_path(local_path)},
            )

        await asyncio.get_running_loop().run_in_executor(None, _sync_upload)
        url = self._public_url(target_bucket, destination_key)
        logger.info("Uploaded %s to %s", local_path, url)
        return url

    async def upload_and_cleanup(
        self,
        file_path: str,
        destination_key: str,
        *,
        bucket_name: Optional[str] = None,
        cleanup: bool = True,
    ) -> str:
        url = await self.upload_file(file_path, destination_key, bucket_name=bucket_name)
        if cleanup:
            try:
                os.remove(file_path)
            except OSError as e:
                logger.warning("Failed to remove temporary file %s: %s", file_path, e)
        return url


# Global instance
cloud_storage = CloudStorage()
