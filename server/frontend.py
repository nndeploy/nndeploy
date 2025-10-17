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
from typing import Optional, TypedDict, Tuple
from urllib.parse import urlsplit, urlunsplit

import requests
from typing_extensions import NotRequired

# ---------------------------- Constants ----------------------------

REQUEST_TIMEOUT = 60  # seconds
FRONTEND_ROOT = Path.cwd() / "frontend"

# Used when caller passes the default bang-string "!" (auto resolve via config)
DEFAULT_PROVIDER = ("nndeploy", "nndeploy_frontend", "v1.4.0")
DEFAULT_VERSION_STRING = "!"

GITHUB_HOST = "github.com"
GITEE_HOST = "gitee.com"
GITHUB_API_BASE = "https://api.github.com"


# ---------------------------- Typed dicts ----------------------------

class Asset(TypedDict):
    name: str
    browser_download_url: str


class Release(TypedDict):
    tag_name: str
    prerelease: bool
    assets: NotRequired[list[Asset]]


# ---------------------------- Provider ----------------------------

@dataclass
class FrontEndProvider:
    owner: str
    repo: str

    @property
    def _base(self) -> str:
        return f"{GITHUB_API_BASE}/repos/{self.owner}/{self.repo}/releases"

    @cached_property
    def latest(self) -> Release:
        r = requests.get(f"{self._base}/latest", timeout=REQUEST_TIMEOUT)
        if r.status_code == 404:
            # Some repos don't expose /latest; fall back to the releases list
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
            # Fallback: iterate all releases and match tag_name
            r = requests.get(self._base, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            for rel in r.json():
                if rel["tag_name"] in (ver, f"v{ver}"):
                    return rel
            raise RuntimeError(f"Release {ver} not found")
        r.raise_for_status()
        return r.json()


# ---------------------------- Download helpers ----------------------------

def _swap_host(url: str, new_host: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, new_host, parts.path, parts.query, parts.fragment))


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


def _download_with_fallbacks(urls: list[str], dest: Path) -> None:
    last_err: Optional[Exception] = None
    for u in urls:
        try:
            logging.info("download %s → %s", u, dest)
            _stream_download(u, dest)
            return
        except Exception as e:
            logging.warning("download failed: %s", e)
            last_err = e
    raise RuntimeError(f"all download attempts failed; last error: {last_err}")


def _download_via_release(rel: Release, dest: Path, asset_name: str = "dist.zip") -> None:
    """
    Download via GitHub API release object.
    Priority: exact 'asset_name' → single .zip asset → error.
    """
    assets = rel.get("assets", []) or []
    asset = next((a for a in assets if a["name"] == asset_name), None)
    if asset is None:
        zips = [a for a in assets if a["name"].lower().endswith(".zip")]
        if len(zips) == 1:
            asset = zips[0]
        else:
            names = [a["name"] for a in assets]
            raise RuntimeError(f"asset '{asset_name}' not found; available assets: {names}")
    browser_url = asset["browser_download_url"]

    try_urls = []
    try:
        try_urls.append(_swap_host(browser_url, GITEE_HOST))
    except Exception:
        pass
    try_urls.append(browser_url)
    _download_with_fallbacks(try_urls, dest)


# ---------------------------- Config + version utils ----------------------------

def _semver_tuple(v: str) -> Tuple[int, int, int]:
    """
    Parse 'vX.Y.Z[-suffix]' into (X, Y, Z); non-parsable parts become 0.
    """
    v = v.strip().lstrip("v")
    core = re.split(r"[-+]", v, 1)[0]
    parts = core.split(".")
    out = []
    for i in range(3):
        try:
            out.append(int(parts[i]))
        except Exception:
            out.append(0)
    return tuple(out)  # type: ignore[return-value]


def _cmp_ver(a: str, b: str) -> int:
    ta, tb = _semver_tuple(a), _semver_tuple(b)
    return (ta > tb) - (ta < tb)


def _match_constraint(nndeploy_ver: str, expr: str) -> bool:
    """
    Support constraint expressions like '>=0.2.12,<0.3.0' or '>=0.3.0'.
    All comma-separated conditions must hold.
    """
    nv = nndeploy_ver
    for cond in [c.strip() for c in expr.split(",") if c.strip()]:
        m = re.match(r"(>=|<=|>|<|==)?\s*v?(\d+\.\d+\.\d+)", cond)
        if not m:
            return False
        op, ver = m.groups()
        op = op or "=="
        cmpres = _cmp_ver(nv, ver)
        ok = {
            "==": cmpres == 0,
            ">=": cmpres >= 0,
            "<=": cmpres <= 0,
            ">":  cmpres > 0,
            "<":  cmpres < 0,
        }[op]
        if not ok:
            return False
    return True


def _load_versions_config() -> dict:
    """
    Import CONFIG from a sibling 'version.py'.
    If running inside a package (python -m pkg.module), a relative import is also attempted.
    """
    CONFIG = None
    try:
        # Script-style import (same directory, not a package)
        from version import CONFIG as _CONF  # type: ignore
        CONFIG = _CONF
    except Exception:
        try:
            # Package-style relative import
            from .version import CONFIG as _CONF  # type: ignore
            CONFIG = _CONF
        except Exception as e:
            raise RuntimeError("version.py not found or import failed") from e

    if not isinstance(CONFIG, dict):
        raise RuntimeError("CONFIG not found or not a dict in version.py")

    if CONFIG.get("schema") != 1:
        raise RuntimeError("version CONFIG schema unsupported or missing (expect 1)")
    return CONFIG


def _normalize_version(ver: Optional[str]) -> Optional[str]:
    """
    Normalize nndeploy.__version__ into a plain 'X.Y.Z' string.
    Handles cases like 'nndeploy 2.6.1' → '2.6.1'.
    """
    if not ver:
        return None
    parts = ver.strip().split()
    if len(parts) > 1 and re.match(r"\d+\.\d+\.\d+", parts[-1]):
        return parts[-1]
    return ver.strip()


def _normalize_frontend_entry(x: object) -> dict:
    """
    Normalize various frontend config shapes into {'tag': str, 'asset': Optional[str]}.
    Supported inputs:
      - "v1.4.0"
      - {"frontend": "v1.4.0", "asset": "..."}
      - {"frontend": {"tag": "v1.4.0", "asset": "..."}}
      - {"tag": "v1.4.0", "asset": "..."}
    """
    if x is None:
        return {}
    if isinstance(x, str):
        return {"tag": x}
    if isinstance(x, dict):
        # Case: {"frontend": "v1.4.0", "asset": "..."}
        if isinstance(x.get("frontend"), str):
            return {"tag": x["frontend"], "asset": x.get("asset")}
        # Case: {"frontend": {"tag": "...", "asset": "..."}} (+ optional top-level asset)
        if isinstance(x.get("frontend"), dict):
            fe = x["frontend"]
            return {"tag": fe.get("tag") or fe.get("frontend"), "asset": fe.get("asset") or x.get("asset")}
        # Case: already normalized child: {"tag": "...", "asset": "..."}
        if "tag" in x or "asset" in x:
            return {"tag": x.get("tag"), "asset": x.get("asset")}
    return {}


def _resolve_frontend_from_config() -> tuple[str, str, str, str]:
    """
    Return (owner, repo, tag, asset) for the frontend bundle.
    Resolution order: exact versions → range rules → fallback.
    """
    cfg = _load_versions_config()

    # Provider (no asset here). Allow either flat default_provider or sectioned default_provider["frontend"].
    defprov = (cfg.get("default_provider") or {}).get("frontend") or cfg.get("default_provider") or {}
    owner = defprov.get("owner") or DEFAULT_PROVIDER[0]
    repo = defprov.get("repo") or DEFAULT_PROVIDER[1]

    # Resolve nndeploy version (normalize if it contains a prefix).
    nndeploy_ver: Optional[str] = None
    try:
        import nndeploy  # type: ignore
        raw_ver = getattr(nndeploy, "__version__", None)
        nndeploy_ver = _normalize_version(raw_ver)
    except Exception:
        pass

    chosen: dict = {}
    if nndeploy_ver:
        # Exact table: CONFIG["versions"][nndeploy_ver]
        vermap = cfg.get("versions", {}) or {}
        hit = vermap.get(nndeploy_ver)
        if hit is not None:
            fe = hit.get("frontend", hit) if isinstance(hit, dict) else hit
            chosen = _normalize_frontend_entry(fe)

        # Range rules: first matching rule wins
        if not chosen:
            for r in cfg.get("ranges", []) or []:
                expr = r.get("nndeploy")
                if expr and _match_constraint(nndeploy_ver, expr):
                    fe = r.get("frontend", r)
                    chosen = _normalize_frontend_entry(fe)
                    if chosen:
                        break

    # Fallback block
    if not chosen:
        fb = cfg.get("fallback") or {}
        fe = fb.get("frontend", fb)
        chosen = _normalize_frontend_entry(fe)

    tag = chosen.get("tag") or chosen.get("frontend") or DEFAULT_PROVIDER[2]
    asset = chosen.get("asset") or "dist.zip"
    return owner, repo, str(tag), str(asset)


# ---------------------------- Public manager ----------------------------

class FrontendManager:
    """
    Resolve and download the correct frontend bundle (dist.zip or a configured asset)
    based on the running nndeploy version and the mapping in version.py::CONFIG.
    """
    VERSION_RE = re.compile(r"^([\w-]+)/([\w_.-]+)@(v?\d+\.\d+\.\d+|latest)$")

    @classmethod
    def init_frontend(cls, version_string: str = DEFAULT_VERSION_STRING) -> str:
        """
        Entry point. Creates cache root and dispatches to implementation.
        - version_string = "!" → auto resolve via CONFIG and nndeploy.__version__
        - version_string = "owner/repo@vX.Y.Z" → direct fetch of that release tag
        """
        FRONTEND_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            return cls._impl(version_string)
        except Exception as exc:
            logging.error("init front end failed: %s", exc, exc_info=True)
            sys.exit(1)

    @classmethod
    def _impl(cls, ver_str: str) -> str:
        # Resolve owner/repo/tag/asset either from CONFIG ("!") or explicit "owner/repo@tag"
        if ver_str == DEFAULT_VERSION_STRING:
            owner, repo, tag, asset = _resolve_frontend_from_config()
        else:
            m = cls.VERSION_RE.match(ver_str)
            if m is None:
                raise argparse.ArgumentTypeError(f"illegal string: {ver_str}")
            owner, repo, tag = m.groups()
            asset = "dist.zip"  # default asset when caller specifies an explicit tag

        # Direct download path for non-latest tags
        if tag != "latest":
            dest = FRONTEND_ROOT / f"{owner}_{repo}" / tag.lstrip("v")
            dist_dir = dest / "dist"
            if dist_dir.exists():
                logging.info("use cached front end: %s", dist_dir)
                return str(dest)

            gitee_url = f"https://{GITEE_HOST}/{owner}/{repo}/releases/download/{tag}/{asset}"
            github_url = f"https://{GITHUB_HOST}/{owner}/{repo}/releases/download/{tag}/{asset}"

            dest.mkdir(parents=True, exist_ok=True)
            try:
                _download_with_fallbacks([gitee_url, github_url], dest)
                return str(dest)
            except Exception as err:
                # Best-effort cleanup for empty dest on failure
                try:
                    dest.rmdir()
                except Exception as e:
                    logging.debug("cleanup empty dir failed: %s", e)
                logging.warning("direct download attempts failed: %s, rolling up to API", err)

        # API fallback (latest or when direct download failed)
        provider = FrontEndProvider(owner, repo)
        rel = provider.latest if tag == "latest" else provider.by_tag(tag)
        semver = rel["tag_name"].lstrip("v")
        dest = FRONTEND_ROOT / f"{owner}_{repo}" / semver
        if not dest.exists():
            logging.info("API download front end assets %s/%s@%s → %s", owner, repo, semver, dest)
            dest.mkdir(parents=True, exist_ok=True)
            _download_via_release(rel, dest, asset_name=asset)
        return str(dest)

    @staticmethod
    def templates_path() -> Optional[str]:
        return None

    @staticmethod
    def embedded_docs_path() -> Optional[str]:
        return None
