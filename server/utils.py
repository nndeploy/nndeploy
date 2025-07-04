import os
from typing import Any, Dict, List, Optional

def extract_encode_output_path(task_json: Dict[str, Any],
                               resources_root: str) -> Optional[str]:
    def _search(nodes: List[Dict[str, Any]]) -> Optional[str]:
        for node in nodes:
            if node.get("key_") == "nndeploy::codec::OpenCvImagesEncode":
                raw_path = node.get("path_")
                if not raw_path:
                    return None
                rel_path = os.path.relpath(raw_path, resources_root)
                return rel_path
            sub_nodes = node.get("node_repository_", [])
            if sub_nodes:
                found = _search(sub_nodes)
                if found:
                    return found
        return None

    return _search(task_json.get("node_repository_", []))