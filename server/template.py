import logging
import sys
import tempfile
import zipfile
import requests
from pathlib import Path
from typing import Optional
import re
import time
from functools import cached_property
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

# HTTP request timeout in seconds
REQUEST_TIMEOUT = 60

# Root directory for cached templates
TEMPLATE_ROOT = Path.cwd() / "resources" / "template"

# Default GitHub provider (owner, repo, tag)
DEFAULT_PROVIDER = ("nndeploy", "nndeploy-workflow", "v1.0.0")

# Default version string, used for fallback
DEFAULT_VERSION_STRING = "!"

GITHUB_HOST = "github.com"
GITEE_HOST = "gitee.com"
GITHUB_API_BASE = "https://api.github.com"

class Asset(dict):
    name: str
    browser_download_url: str

@dataclass
class TemplateProvider:
    owner: str
    repo: str

    @property
    def _base(self) -> str:
        return f"{GITHUB_API_BASE}/repos/{self.owner}/{self.repo}/releases"

    @cached_property
    def latest(self) -> dict:
        r = requests.get(f"{self._base}/latest", timeout=REQUEST_TIMEOUT)
        if r.status_code == 404:
            r = requests.get(self._base, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            releases = r.json()
            if not releases:
                raise RuntimeError("No releases found on GitHub")
            return releases[0]
        r.raise_for_status()
        return r.json()

    def by_tag(self, ver: str) -> dict:
        r = requests.get(f"{self._base}/tags/{ver}", timeout=REQUEST_TIMEOUT)
        if r.status_code == 404:
            r = requests.get(self._base, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            for rel in r.json():
                if rel["tag_name"] in (ver, f"v{ver}"):
                    return rel
            raise RuntimeError(f"Release {ver} not found on GitHub")
        r.raise_for_status()
        return r.json()

def _swap_host(url: str, new_host: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, new_host, parts.path, parts.query, parts.fragment))

def _stream_download_with_retry(url: str, dest_dir: Path, retries: int = 3, delay: float = 2.0) -> None:
    for attempt in range(1, retries + 1):
        try:
            logging.info(f"[Attempt {attempt}] Downloading from {url}")
            with tempfile.TemporaryFile() as tmp:
                r = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                for chunk in r.iter_content(8192):
                    if chunk:
                        tmp.write(chunk)
                tmp.seek(0)
                with zipfile.ZipFile(tmp) as zf:
                    zf.extractall(dest_dir)
            logging.info(f"Template extracted successfully to {dest_dir}")
            return
        except Exception as e:
            logging.warning(f"Download failed on attempt {attempt}: {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                logging.error(f"All {retries} download attempts failed for {url}")
                raise

def _download_with_fallbacks(urls: list[str], dest: Path, retries_each: int = 3, delay: float = 2.0) -> None:
    last_err: Optional[Exception] = None
    for u in urls:
        try:
            _stream_download_with_retry(u, dest, retries=retries_each, delay=delay)
            return
        except Exception as e:
            last_err = e
            logging.warning(f"Download attempt failed for {u}: {e}")
    raise RuntimeError(f"All download attempts failed. Last error: {last_err}")

def _download_via_release(rel: dict, dest: Path) -> None:
    asset_url = next((a["browser_download_url"] for a in rel.get("assets", []) if a["name"] == "nndeploy-workflow.zip"), None)
    if asset_url is None:
        raise RuntimeError("nndeploy-workflow.zip not found in release assets")
    try_urls = []
    try:
        try_urls.append(_swap_host(asset_url, GITEE_HOST))
    except Exception:
        pass
    try_urls.append(asset_url)
    _download_with_fallbacks(try_urls, dest)

class WorkflowTemplateManager:
    VERSION_RE = re.compile(r"^([\w-]+)/([\w_.-]+)@(v?\d+\.\d+\.\d+|latest)$")

    @classmethod
    def init_templates(cls, version_string: str = DEFAULT_VERSION_STRING) -> Optional[str]:
        TEMPLATE_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            return cls._impl(version_string)
        except Exception as exc:
            logging.error("Failed to initialize workflow templates: %s", exc, exc_info=True)
            return None

    @classmethod
    def _impl(cls, ver_str: str) -> str:
        if ver_str == DEFAULT_VERSION_STRING:
            owner, repo, tag = DEFAULT_PROVIDER
        else:
            m = cls.VERSION_RE.match(ver_str)
            if m is None:
                raise ValueError(f"Invalid version string format: {ver_str}")
            owner, repo, tag = m.groups()

        if tag != "latest":
            dest = TEMPLATE_ROOT
            dest_inner = dest / "nndeploy-workflow"
            if dest_inner.exists():
                logging.info(f"use cached templates at {dest_inner}")
                return str(dest)

            gitee_url = f"https://{GITEE_HOST}/{owner}/{repo}/releases/download/{tag}/nndeploy-workflow.zip"
            github_url = f"https://{GITHUB_HOST}/{owner}/{repo}/releases/download/{tag}/nndeploy-workflow.zip"

            try:
                logging.info(f"Attempting direct download (Gitee→GitHub): {gitee_url} / {github_url} → {dest}")
                dest.mkdir(parents=True, exist_ok=True)
                _download_with_fallbacks([gitee_url, github_url], dest)
                return str(dest)
            except Exception as err:
                try:
                    dest.rmdir()
                except Exception as e:
                    logging.debug(f"Failed to clean up failed directory: {e}")
                logging.warning(f"Direct download failed. Falling back to GitHub API: {err}")

        provider = TemplateProvider(owner, repo)
        rel = provider.latest if tag == "latest" else provider.by_tag(tag)
        semver = rel["tag_name"].lstrip("v")
        dest = TEMPLATE_ROOT
        if not dest.exists():
            logging.info(f"Downloading templates via API: {owner}/{repo}@{semver} → {dest}")
            dest.mkdir(parents=True, exist_ok=True)
            _download_via_release(rel, dest)

        return str(dest)
