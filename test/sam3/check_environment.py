"""Check the installed SAM3/nndeploy environment without loading model weights."""

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Run small CUDA/cuDNN operations")
    parser.add_argument("--server", action="store_true", help="Exercise the existing Server with a NumPy Graph")
    parser.add_argument("--output", type=Path, help="Write the JSON report outside the source tree")
    args = parser.parse_args()
    report = {"python": sys.version, "executable": sys.executable, "checks": {}}

    def check(name, action):
        try:
            report["checks"][name] = {"ok": True, "details": action()}
        except Exception:
            report["checks"][name] = {"ok": False, "error": traceback.format_exc()}
        print(name, "PASS" if report["checks"][name]["ok"] else "FAIL", flush=True)

    def imports():
        names = ["nndeploy", "nndeploy.server", "numpy", "cv2", "torch",
                 "torchvision", "triton", "timm", "scipy", "skimage",
                 "pycocotools.mask", "fastapi", "uvicorn",
                 "nndeploy.segment.sam3.model_builder"]
        locations = {name: importlib.import_module(name).__file__ for name in names}
        assert not any(name == "sam3" or name.startswith("sam3.") for name in sys.modules)
        versions = {name: importlib.metadata.version(name) for name in
                    ["numpy", "opencv-python", "torch", "torchvision", "triton",
                     "timm", "pycocotools", "fastapi", "uvicorn", "nndeploy"]}
        return {"locations": locations, "versions": versions}

    def arrays_and_codecs():
        import cv2
        import numpy as np
        import torch
        from pycocotools import mask as mask_util
        from nndeploy.device import Tensor

        array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        np.testing.assert_array_equal(torch.from_numpy(array).numpy(), array)
        np.testing.assert_array_equal(np.asarray(Tensor.from_numpy(array)), array)
        mask = np.zeros((17, 23), dtype=np.uint8)
        mask[2:11, 4:19] = 1
        rle = mask_util.encode(np.asfortranarray(mask))
        wire = {"size": rle["size"], "counts": rle["counts"].decode("utf-8")}
        np.testing.assert_array_equal(mask_util.decode(json.loads(json.dumps(wire))), mask)
        ok, encoded = cv2.imencode(".png", mask * 255)
        assert ok
        np.testing.assert_array_equal(cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE), mask * 255)
        return {"numpy_torch_native_roundtrip": True, "rle_roundtrip": True, "png_roundtrip": True}

    def cuda():
        import torch
        assert torch.cuda.is_available(), "CUDA is unavailable"
        with torch.inference_mode():
            value = torch.ones((32, 32), device="cuda")
            torch.testing.assert_close(value @ value, torch.full_like(value, 32))
            result = torch.nn.functional.conv2d(
                torch.ones((1, 1, 32, 32), device="cuda"),
                torch.ones((1, 1, 3, 3), device="cuda"))
            torch.testing.assert_close(result, torch.full_like(result, 9))
            torch.cuda.synchronize()
        return {"device": torch.cuda.get_device_name(0), "torch_cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version()}

    def server():
        import numpy as np
        import nndeploy.base
        import nndeploy.dag
        from fastapi.testclient import TestClient
        from nndeploy.server import Server

        class ArrayCopy(nndeploy.dag.Node):
            def __init__(self, name, inputs, outputs):
                super().__init__(name, inputs, outputs)
                self.set_key("test.sam3.EnvironmentArrayCopy")
                self.set_input_type(np.ndarray)
                self.set_output_type(np.ndarray)

            def run(self):
                value = self.get_input(0).get(self)
                self.get_output(0).set(np.array(value, copy=True))
                return nndeploy.base.Status.ok()

        def build_graph():
            source, target = nndeploy.dag.Edge("input"), nndeploy.dag.Edge("output")
            graph = nndeploy.dag.Graph("sam3_environment", [source], [target])
            graph.add_node(ArrayCopy("copy", [source], [target]))
            return graph

        field = {"type": "array", "dtype": "float32", "shape": [2, 3]}
        config = {"inputs": {"value": dict(field, edge="input")},
                  "outputs": {"value": dict(field, edge="output")},
                  "execution": {"isolation": "thread"}}
        with tempfile.TemporaryDirectory(prefix="sam3-environment-") as directory:
            instance = Server(graph_factory=build_graph, config=config, data_dir=Path(directory))
            with TestClient(instance.app) as client:
                first_task = None
                for values in ([0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]):
                    body = {"dtype": "float32", "shape": [2, 3], "data": values}
                    response = client.post("/v1/server/default/deploy", json={"inputs": {"value": body}})
                    assert response.status_code == 200, response.text
                    result = response.json()
                    assert result["outputs"]["value"] == body, result
                    if first_task is None:
                        first_task, first_body = result["task_id"], body
                saved = client.get("/v1/tasks/" + first_task)
                assert saved.status_code == 200, saved.text
                assert saved.json()["outputs"]["value"] == first_body
        return {"requests": 2, "graph_array_roundtrip": True, "task_result_retained": True}

    check("imports", imports)
    check("arrays_and_codecs", arrays_and_codecs)
    if args.cuda:
        check("cuda", cuda)
    if args.server:
        check("server_graph", server)
    report["ok"] = all(item["ok"] for item in report["checks"].values())
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
