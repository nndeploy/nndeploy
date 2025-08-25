import logging
import requests
import os
import re
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path
from time import sleep
import shutil
from urllib.parse import urlparse
from modelscope.hub.file_download import model_file_download

def extract_encode_output_paths(task_json: Dict[str, Any]) -> List[Dict[str, str]]:
    preview_target_keys = {
        "nndeploy::codec::OpenCvImageEncode",
        "nndeploy::codec::OpenCvVideoEncode",
    }

    text_target_keys = {
        "nndeploy::qwen::PrintNode"
    }

    preview_path = []
    preview_text = []

    def _search(nodes: List[Dict[str, Any]]):
        for node in nodes:
            if node.get("key_") in preview_target_keys:
                name = node.get("name_", "")
                path = node.get("path_", "")
                if path:
                    preview_path.append({"name": name, "path": path})
            if node.get("key_") in text_target_keys:
                name = node.get("name_", "")
                text = Path(".tmp/output_" + name + ".txt").read_text(encoding="utf-8")
                if text:
                    preview_text.append({"name": name, "text":text})

            sub_nodes = node.get("node_repository_", [])
            if sub_nodes:
                _search(sub_nodes)

    _search(task_json.get("node_repository_", []))
    return preview_path, preview_text

# def _handle_urls(graph_json: Dict, work_dir: str) -> Dict[str, str]:

#     url_fields = {
#         "image_url_": "images",
#         "video_url_": "videos",
#         "audio_url_": "audios",
#         "model_url_": "models",
#         "other_url_": "others"
#     }

#     result_paths = {}

#     def _safe_remove(path: Path):
#         """save delete target dir"""
#         try:
#             if not path.exists():
#                 return
#             if path.is_file() or path.is_symlink():
#                 path.unlink(missing_ok=True)
#             else:
#                 shutil.rmtree(path, ignore_errors=True)
#         except Exception as e:
#             logging.warning(f"Failed to cleanup path {path}: {e}")

#     for field, subdir in url_fields.items():
#         if field not in graph_json:
#             continue

#         url_list = graph_json[field]
#         if not isinstance(url_list, list):
#             logging.warning(f"{field} is not a list")
#             continue
#         if not url_list:
#             logging.info(f"{field} is an empty list, skipping.")
#             continue

#         for idx, url in enumerate(url_list):
#             if not isinstance(url, str):
#                 logging.warning(f"{field}.{idx} is not a string.")
#                 continue

#             try:
#                 type_, uri = url.split("@", 1)
#             except ValueError:
#                 logging.warning(f"Invalid URL format in {field}.{idx}: {url}")
#                 continue

#             save_dir = Path(work_dir) / subdir
#             save_dir.mkdir(parents=True, exist_ok=True)

#             local_path = None

#             # === Handle modelscope ===
#             if type_ == "modelscope":
#                 expected_dest: Path | None = None
#                 try:
#                     model_id, file_path = uri.split(":", 1)
#                     expected_dest = (save_dir / file_path).resolve()
#                     local_path = model_file_download(
#                         model_id=model_id,
#                         file_path=file_path,
#                         local_dir=save_dir
#                     )
#                 except Exception as e:
#                     logging.error(f"Failed to download modelscope model for {field}.{idx}: {e}")
#                     if field == "model_url_" and expected_dest is not None:
#                         _safe_remove(expected_dest)
#                     continue

#             # === Handle http/https ===
#             elif type_ in ("http", "https"):
#                 try:
#                     parsed = urlparse(uri)
#                     filename = os.path.basename(parsed.path) or "downloaded_file"
#                     save_path = save_dir / filename

#                     if not save_path.exists():
#                         logging.info(f"Downloading {uri} to {save_path}")
#                         r = requests.get(uri, stream=True, timeout=10)
#                         r.raise_for_status()
#                         with open(save_path, "wb") as f:
#                             for chunk in r.iter_content(chunk_size=8192):
#                                 f.write(chunk)
#                         logging.info(f"Saved to {save_path}")
#                     else:
#                         logging.info(f"File already exists: {save_path}")

#                     local_path = save_path

#                 except Exception as e:
#                     logging.error(f"Failed to download http resource for {field}.{idx}: {e}")
#                     if field == "model_url_" and save_path is not None:
#                         _safe_remove(save_path)
#                     continue
#             elif type_.startswith("template"):
#                     logging.info(f"Skip processing template resource for {field}.{idx}")
#                     continue
#             else:
#                 logging.warning(f"Unsupported or failed to download resource for {field}.{idx} with type '{type_}'")

#             # === Store result ===
#             if local_path:
#                 result_paths[f"{field}.{idx}"] = str(local_path)

#     return result_paths

def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    return name.rstrip(" .")


def _ensure_under(base: Path, target: Path) -> bool:
    try:
        base = base.resolve()
        target = target.resolve()
        return str(target).startswith(str(base) + os.sep)
    except Exception:
        return False


def _safe_remove(path: Path):
    """safe deleter"""
    try:
        if not path.exists():
            return
        if path.is_file() or path.is_symlink():
            try:
                path.unlink(missing_ok=True)
            except TypeError:
                if path.exists():
                    path.unlink()
        else:
            shutil.rmtree(path, ignore_errors=True)
    except Exception as e:
        logging.warning(f"Failed to cleanup path {path}: {e}")


def _atomic_write_from_stream(response: requests.Response, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(dst_path.parent), delete=False) as tf:
        tmp_name = tf.name
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # avoid keep-alive empty blank
                tf.write(chunk)
    # atomic replace（Windows/Linux/macOS）
    os.replace(tmp_name, dst_path)


def _download_with_retries(uri: str, save_path: Path, retries: int = 3, timeout=(10, 60)):
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            logging.info(f"Downloading {uri} to {save_path} (attempt {attempt}/{retries})")
            r = requests.get(uri, stream=True, timeout=timeout)
            r.raise_for_status()
            _atomic_write_from_stream(r, save_path)
            logging.info(f"Saved to {save_path}")
            return save_path
        except Exception as e:
            last_err = e
            logging.warning(f"Download failed (attempt {attempt}/{retries}) for {uri}: {e}")
            _safe_remove(save_path)
            if attempt < retries:
                sleep(1.5 * attempt)
    assert last_err is not None
    raise last_err


def _handle_urls(graph_json: Dict, work_dir: str) -> Dict[str, str]:
    """
    process graph_json *_url_ attributes:
    - modelscope/http/https download
    - faild clean(仅对 model_url_)
    - return { "field.idx": "file absolute path" }
    """
    url_fields = {
        "image_url_": "images",
        "video_url_": "videos",
        "audio_url_": "audios",
        "model_url_": "models",
        "other_url_": "others"
    }

    result_paths: Dict[str, str] = {}

    for field, subdir in url_fields.items():
        if field not in graph_json:
            continue

        url_list = graph_json[field]
        if not isinstance(url_list, list):
            logging.warning(f"{field} is not a list")
            continue
        if not url_list:
            logging.info(f"{field} is an empty list, skipping.")
            continue

        for idx, url in enumerate(url_list):
            if not isinstance(url, str):
                logging.warning(f"{field}.{idx} is not a string.")
                continue

            try:
                type_, uri = url.split("@", 1)
            except ValueError:
                logging.warning(f"Invalid URL format in {field}.{idx}: {url}")
                continue

            save_dir = Path(work_dir) / subdir
            save_dir.mkdir(parents=True, exist_ok=True)

            local_path: Optional[Path] = None

            # === Handle modelscope ===
            if type_ == "modelscope":
                expected_dest: Optional[Path] = None
                try:
                    model_id, file_path = uri.split(":", 1)
                    rel_path = Path(file_path)
                    expected_dest = (save_dir / rel_path).resolve()
                    if not _ensure_under(save_dir, expected_dest):
                        logging.error(f"Path traversal detected for {field}.{idx}: {expected_dest}")
                        continue

                    local_path = model_file_download(
                        model_id=model_id,
                        file_path=str(rel_path),
                        local_dir=str(save_dir)
                    )
                    local_path = Path(local_path).resolve()

                    if not _ensure_under(save_dir, local_path):
                        logging.error(f"Downloaded path escapes save_dir for {field}.{idx}: {local_path}")
                        _safe_remove(local_path)
                        local_path = None
                        continue

                except Exception as e:
                    logging.error(f"Failed to download modelscope model for {field}.{idx}: {e}")
                    if field == "model_url__":
                        if expected_dest is not None:
                            _safe_remove(expected_dest)
                        if local_path is not None:
                            _safe_remove(local_path)
                    continue

            # === Handle http/https ===
            elif type_ in ("http", "https"):
                try:
                    parsed = urlparse(uri)
                    filename = _sanitize_filename(os.path.basename(parsed.path) or "downloaded_file")
                    save_path = (save_dir / filename).resolve()

                    if not _ensure_under(save_dir, save_path):
                        logging.error(f"Path traversal detected for {field}.{idx}: {save_path}")
                        continue

                    if save_path.exists():
                        logging.info(f"File already exists: {save_path}")
                        local_path = save_path
                    else:
                        local_path = _download_with_retries(uri, save_path)

                except Exception as e:
                    logging.error(f"Failed to download http resource for {field}.{idx}: {e}")
                    if field == "model_url_":
                        try:
                            _safe_remove(save_path)  # type: ignore[name-defined]
                        except Exception:
                            pass
                    continue

            # === Handle template:* ===
            elif type_.startswith("template"):
                logging.info(f"Skip processing template resource for {field}.{idx}")
                continue

            else:
                logging.warning(f"Unsupported or failed to download resource for {field}.{idx} with type '{type_}'")
                continue

            # === Store result ===
            if local_path:
                result_paths[f"{field}.{idx}"] = str(local_path)

    return result_paths
