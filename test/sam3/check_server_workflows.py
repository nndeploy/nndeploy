"""Exercise saved SAM3 workflows through the unmodified nndeploy Server."""

import argparse
import json
from pathlib import Path
import tempfile

from fastapi.testclient import TestClient
import torch

from nndeploy.server import Server
from nndeploy.segment.sam3 import ModelConfig, RuntimeOwner
from nndeploy.segment.sam3.graphs import SAM3ImageConceptGraph, SAM3ImageInteractiveGraph
from nndeploy.segment.sam3.modules import Modules
from nndeploy.segment.sam3.workflow import server_factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output); destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    modules = Modules()
    with RuntimeOwner(ModelConfig(args.checkpoint, args.device), destination / "results", modules) as owner:
        config = {"inputs": {"request": {"type": "json", "edge": "request"}},
                  "outputs": {"result": {"type": "json", "edge": "result"}},
                  "execution": {"isolation": "thread"}}
        for kind, cls in (("concept", SAM3ImageConceptGraph), ("interact", SAM3ImageInteractiveGraph)):
            cls(kind, owner=owner).save_file(str(destination / f"{kind}.json"))
        with tempfile.TemporaryDirectory(prefix="sam3-server-") as directory:
            instance = Server(name="concept", graph_factory=server_factory(destination / "concept.json", owner), config=config, data_dir=Path(directory))
            instance.add(name="interact", graph_factory=server_factory(destination / "interact.json", owner), config=config)
            with TestClient(instance.app) as client:
                tasks = []
                def invoke(kind, request):
                    response = client.post(f"/v1/server/{kind}/deploy", json={"inputs": {"request": request}})
                    assert response.status_code == 200, response.text
                    body = response.json(); tasks.append(body["task_id"])
                    return body["outputs"]["result"]
                created = invoke("concept", {"op": "create", "image": args.image})
                assert created["ok"], created
                context_id = created["context_id"]
                result = invoke("concept", {"context_id": context_id, "text": "truck"})
                assert result["ok"] and result["instances"], result
                refined = invoke("interact", {"context_id": context_id, "groups": [{"selection": {"result_id": result["result_id"], "index": 0}, "multimask": False}]})
                assert refined["ok"] and refined["groups"][0]["candidates"], refined
                history = client.get("/v1/tasks/" + tasks[0])
                assert history.status_code == 200 and history.json()["outputs"]["result"] == created
                assert modules.calls["vision_encoder"] == 1, modules.snapshot()
                released = invoke("interact", {"op": "release", "context_id": context_id})
                assert released["ok"]
                expired = invoke("concept", {"context_id": context_id, "text": "truck"})
                assert not expired["ok"] and expired["error"]["code"] == "unknown_context"
    report = {"server_saved_workflows": True, "shared_image_runtime": True, "vision_encoder_calls": modules.calls["vision_encoder"], "requests": len(tasks), "history_snapshot": True, "business_api_compatibility_tested": False}
    (destination / "report.json").write_text(json.dumps(report, indent=2))
    print("PASS", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
