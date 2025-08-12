# executor.py

import requests
import os
import logging
import json
from typing import Dict, Tuple, Any, List
from pathlib import Path
from urllib.parse import urlparse

from modelscope.hub.file_download import model_file_download

from nndeploy.dag import GraphRunner

class GraphExecutor:
    """Encapsulate nndeploy load and run logic"""
    def __init__(self, resources, cache_type=False, cache_size=None):
        self.runner = GraphRunner()
        self.resources = resources

    def execute(self, graph_json: Dict, task_id: str) -> Tuple[Dict, float]:
        name = graph_json.get("name_")
        if isinstance(graph_json, (dict, list)):
            graph_json = json.dumps(graph_json, ensure_ascii=False)

        return self.runner.run(graph_json, name, task_id)

    def handle_urls(self, graph_json: Dict) -> Dict[str, str]:

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

                save_dir = Path(self.resources) / subdir
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
