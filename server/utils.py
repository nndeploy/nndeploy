from typing import Dict, Any, List

def extract_encode_output_paths(task_json: Dict[str, Any]) -> List[Dict[str, str]]:
    target_keys = {
        "nndeploy::codec::OpenCvImageEncode",
        "nndeploy::codec::OpenCvVideoEncode",
    }

    results = []

    def _search(nodes: List[Dict[str, Any]]):
        for node in nodes:
            if node.get("key_") in target_keys:
                name = node.get("name_", "")
                path = node.get("path_", "")
                if path:
                    results.append({"name": name, "path": path})
            sub_nodes = node.get("node_repository_", [])
            if sub_nodes:
                _search(sub_nodes)

    _search(task_json.get("node_repository_", []))
    return results
