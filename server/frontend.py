from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import zipfile
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional, TypedDict

import requests
from typing_extensions import NotRequired

REQUEST_TIMEOUT = 60  # second
FRONTEND_ROOT = Path.cwd() / "frontend"

DEFAULT_PROVIDER = ("nndeploy", "nndeploy_frontend", "v1.3.0")
DEFAULT_VERSION_STRING = "!"

class Asset(TypedDict):
    name: str
    browser_download_url: str

class Release(TypedDict):
    tag_name: str
    prerelease: bool
    assets: NotRequired[list[Asset]]

@dataclass
class FrontEndProvider:
    owner: str
    repo: str

    @property
    def _base(self) -> str:
        return f"https://api.github.com/repos/{self.owner}/{self.repo}/releases"

    @cached_property
    def latest(self) -> Release:
        r = requests.get(f"{self._base}/latest", timeout=REQUEST_TIMEOUT)
        if r.status_code == 404:
            r = requests.get(self._base, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            releases: list[Release] = r.json()
            if not releases:
                raise RuntimeError("no releases found")
            return releases[0]
        r.raise_for_status()
        return r.json()

    def by_tag(self, ver: str) -> Release:
        r = requests.get(f"{self._base}/tags/{ver}", timeout=REQUEST_TIMEOUT)
        if r.status_code == 404:
            r = requests.get(self._base, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            for rel in r.json():
                if rel["tag_name"] in (ver, f"v{ver}"):
                    return rel
            raise RuntimeError(f"Release {ver} not found")
        r.raise_for_status()
        return r.json()

def _stream_download(url: str, dest_dir: Path) -> None:
    with tempfile.TemporaryFile() as tmp:
        r = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        for chunk in r.iter_content(8192):
            if chunk:
                tmp.write(chunk)
        tmp.seek(0)
        with zipfile.ZipFile(tmp) as zf:
            zf.extractall(dest_dir)


def _download_via_release(rel: Release, dest: Path) -> None:
    asset_url = next((a["browser_download_url"] for a in rel.get("assets", []) if a["name"] == "dist.zip"), None)
    if asset_url is None:
        raise RuntimeError("dist.zip asset missing in release")
    _stream_download(asset_url, dest)

class FrontendManager:
    VERSION_RE = re.compile(r"^([\w-]+)/([\w_.-]+)@(v?\d+\.\d+\.\d+|latest)$")

    @classmethod
    def init_frontend(cls, version_string: str = DEFAULT_VERSION_STRING) -> str:
        FRONTEND_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            return cls._impl(version_string)
        except Exception as exc:
            logging.error("init front end failed: %s", exc, exc_info=True)
            sys.exit(1)

    # ─── implementation ───
    @classmethod
    def _impl(cls, ver_str: str) -> str:
        if ver_str == DEFAULT_VERSION_STRING:
            owner, repo, tag = DEFAULT_PROVIDER
        else:
            m = cls.VERSION_RE.match(ver_str)
            if m is None:
                raise argparse.ArgumentTypeError(f"illegal string: {ver_str}")
            owner, repo, tag = m.groups()

        # cache path
        if tag != "latest":
            dest = FRONTEND_ROOT / f"{owner}_{repo}" / tag.lstrip("v")
            dist_dir = dest / "dist"
            if dist_dir.exists():
                logging.info("use cached front end: %s", dist_dir)
                return str(dest)

            direct_url = f"https://github.com/{owner}/{repo}/releases/download/{tag}/dist.zip"
            try:
                logging.info("direct download %s → %s", direct_url, dest)
                dest.mkdir(parents=True, exist_ok=True)
                _stream_download(direct_url, dest)
                return str(dest)
            except Exception as err:
                try:
                    dest.rmdir()
                except Exception as e:
                    logging.debug("cleanup empty dir failed: %s", e)
                logging.warning("download failed: %s, roll up to API", err)

        provider = FrontEndProvider(owner, repo)
        rel = provider.latest if tag == "latest" else provider.by_tag(tag)
        semver = rel["tag_name"].lstrip("v")
        dest = FRONTEND_ROOT / f"{owner}_{repo}" / semver
        if not dest.exists():
            logging.info("API download front end assets %s/%s@%s → %s", owner, repo, semver, dest)
            dest.mkdir(parents=True, exist_ok=True)
            _download_via_release(rel, dest)
        return str(dest)

    @staticmethod
    def templates_path() -> Optional[str]:
        return None

    @staticmethod
    def embedded_docs_path() -> Optional[str]:
        return None
