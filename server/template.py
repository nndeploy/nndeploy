import logging
import sys
import tempfile
import zipfile
import requests
from pathlib import Path
from typing import Optional, Tuple
import re
import time
from functools import cached_property
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

# ---------------------------- Constants ----------------------------

REQUEST_TIMEOUT = 60  # seconds
TEMPLATE_ROOT = Path.cwd() / "resources" / "template"

# Fallback provider (owner, repo, tag) when config is missing
DEFAULT_PROVIDER = ("nndeploy", "nndeploy-workflow", "v1.0.0")

# "!" means: resolve via version.py::CONFIG + nndeploy.__version__
DEFAULT_VERSION_STRING = "!"

GITHUB_HOST = "github.com"
GITEE_HOST = "gitee.com"
GITHUB_API_BASE = "https://api.github.com"


# ---------------------------- Provider ----------------------------

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


# ---------------------------- Download helpers ----------------------------

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


def _download_via_release(rel: dict, dest: Path, asset_name: str = "nndeploy-workflow.zip") -> None:
    """
    Download via GitHub API release object.
    Priority: exact 'asset_name' → single .zip asset → error.
    """
    assets = rel.get("assets", []) or []
    asset = next((a for a in assets if a.get("name") == asset_name), None)
    if asset is None:
        zips = [a for a in assets if str(a.get("name", "")).lower().endswith(".zip")]
        if len(zips) == 1:
            asset = zips[0]
        else:
            names = [a.get("name") for a in assets]
            raise RuntimeError(f"Asset '{asset_name}' not found; available assets: {names}")
    browser_url = asset["browser_download_url"]

    try_urls = []
    try:
        try_urls.append(_swap_host(browser_url, GITEE_HOST))
    except Exception:
        pass
    try_urls.append(browser_url)
    _download_with_fallbacks(try_urls, dest)


# ---------------------------- Config + version utils ----------------------------

def _load_versions_config() -> dict:
    """
    Import CONFIG from sibling 'version.py' (script style),
    or from '.version' when running inside a package.
    """
    CONFIG = None
    try:
        from version import CONFIG as _CONF  # type: ignore
        CONFIG = _CONF
    except Exception:
        try:
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
    Normalize nndeploy.__version__ into 'X.Y.Z'.
    Handles cases like 'nndeploy 2.6.1' → '2.6.1'.
    """
    if not ver:
        return None
    parts = ver.strip().split()
    if len(parts) > 1 and re.match(r"\d+\.\d+\.\d+", parts[-1]):
        return parts[-1]
    return ver.strip()


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


def _normalize_templates_entry(x: object) -> dict:
    """
    Normalize templates config shapes into {'tag': str, 'asset': Optional[str]}.
    Supported inputs:
      - "v1.0.0"
      - {"templates": "v1.0.0", "asset": "..."}
      - {"templates": {"tag": "v1.0.0", "asset": "..."}}
      - {"tag": "v1.0.0", "asset": "..."}
    """
    if x is None:
        return {}
    if isinstance(x, str):
        return {"tag": x}
    if isinstance(x, dict):
        if isinstance(x.get("templates"), str):
            return {"tag": x["templates"], "asset": x.get("asset")}
        if isinstance(x.get("templates"), dict):
            te = x["templates"]
            return {"tag": te.get("tag") or te.get("templates"), "asset": te.get("asset") or x.get("asset")}
        if "tag" in x or "asset" in x:
            return {"tag": x.get("tag"), "asset": x.get("asset")}
    return {}


def _resolve_templates_from_config() -> tuple[str, str, str, str]:
    """
    Resolve (owner, repo, tag, asset) for the workflow templates bundle.
    Resolution order: exact versions → range rules → fallback.
    """
    cfg = _load_versions_config()

    # Allow either flat default_provider or sectioned default_provider["templates"].
    defprov = (cfg.get("default_provider") or {}).get("templates") or cfg.get("default_provider") or {}
    owner = defprov.get("owner") or DEFAULT_PROVIDER[0]
    repo = defprov.get("repo") or DEFAULT_PROVIDER[1]

    nndeploy_ver: Optional[str] = None
    try:
        import nndeploy  # type: ignore
        raw_ver = getattr(nndeploy, "__version__", None)
        nndeploy_ver = _normalize_version(raw_ver)
    except Exception:
        pass

    chosen: dict = {}
    if nndeploy_ver:
        # Exact table
        vermap = cfg.get("versions", {}) or {}
        hit = vermap.get(nndeploy_ver)
        if hit is not None:
            te = hit.get("templates", hit) if isinstance(hit, dict) else hit
            chosen = _normalize_templates_entry(te)

        # Range rules
        if not chosen:
            for r in cfg.get("ranges", []) or []:
                expr = r.get("nndeploy")
                if expr and _match_constraint(nndeploy_ver, expr):
                    te = r.get("templates", r)
                    chosen = _normalize_templates_entry(te)
                    if chosen:
                        break

    # Fallback
    if not chosen:
        fb = cfg.get("fallback") or {}
        te = fb.get("templates", fb)
        chosen = _normalize_templates_entry(te)

    tag = chosen.get("tag") or chosen.get("templates") or DEFAULT_PROVIDER[2]
    asset = chosen.get("asset") or "nndeploy-workflow.zip"
    return owner, repo, str(tag), str(asset)


# ---------------------------- Public manager ----------------------------

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
        # Resolve owner/repo/tag/asset from config ("!") or parse explicit string
        if ver_str == DEFAULT_VERSION_STRING:
            owner, repo, tag, asset = _resolve_templates_from_config()
        else:
            m = cls.VERSION_RE.match(ver_str)
            if m is None:
                raise ValueError(f"Invalid version string format: {ver_str}")
            owner, repo, tag = m.groups()
            asset = "nndeploy-workflow.zip"

        # Direct download for non-latest tags
        if tag != "latest":
            dest = TEMPLATE_ROOT
            dest_inner = dest / "nndeploy-workflow"  # keep your existing cache marker
            if dest_inner.exists():
                logging.info(f"use cached templates at {dest_inner}")
                return str(dest)

            gitee_url = f"https://{GITEE_HOST}/{owner}/{repo}/releases/download/{tag}/{asset}"
            github_url = f"https://{GITHUB_HOST}/{owner}/{repo}/releases/download/{tag}/{asset}"

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

        # API fallback (latest or when direct download failed)
        provider = TemplateProvider(owner, repo)
        rel = provider.latest if tag == "latest" else provider.by_tag(tag)
        semver = rel["tag_name"].lstrip("v")
        # dest = TEMPLATE_ROOT / repo
        dest = TEMPLATE_ROOT
        dest_inner = dest / "nndeploy-workflow"
        if not dest_inner.exists():
            logging.info(f"Downloading templates via API: {owner}/{repo}@{semver} → {dest}")
            dest.mkdir(parents=True, exist_ok=True)
            _download_via_release(rel, dest, asset_name=asset)

        return str(dest)
