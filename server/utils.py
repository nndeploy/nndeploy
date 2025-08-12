import logging
import requests
import os
from typing import Dict, Any, List
from pathlib import Path
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

def _handle_urls(graph_json: Dict, work_dir: str) -> Dict[str, str]:

    url_fields = {
        "image_url_": "images",
        "video_url_": "videos",
        "audio_url_": "audios",
        "model_url_": "models",
        "other_url_": "others"
    }

    result_paths = {}

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

            local_path = None

            # === Handle modelscope ===
            if type_ == "modelscope":
                try:
                    model_id, file_path = uri.split(":", 1)
                    local_path = model_file_download(
                        model_id=model_id,
                        file_path=file_path,
                        local_dir=save_dir
                    )
                except Exception as e:
                    logging.error(f"Failed to download modelscope model for {field}.{idx}: {e}")
                    continue

            # === Handle http/https ===
            elif type_ in ("http", "https"):
                try:
                    parsed = urlparse(uri)
                    filename = os.path.basename(parsed.path) or "downloaded_file"
                    save_path = save_dir / filename

                    if not save_path.exists():
                        logging.info(f"Downloading {uri} to {save_path}")
                        r = requests.get(uri, stream=True, timeout=10)
                        r.raise_for_status()
                        with open(save_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                        logging.info(f"Saved to {save_path}")
                    else:
                        logging.info(f"File already exists: {save_path}")

                    local_path = save_path

                except Exception as e:
                    logging.error(f"Failed to download http resource for {field}.{idx}: {e}")
                    continue
            elif type_.startswith("template"):
                    logging.info(f"Skip processing template resource for {field}.{idx}")
                    continue
            else:
                logging.warning(f"Unsupported or failed to download resource for {field}.{idx} with type '{type_}'")

            # === Store result ===
            if local_path:
                result_paths[f"{field}.{idx}"] = str(local_path)

    return result_paths