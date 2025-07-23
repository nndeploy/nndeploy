from typing import Dict, Any, List
from pathlib import Path

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
