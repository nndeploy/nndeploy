"""CPU result snapshots and versioned, incrementally written video results."""

import json
from pathlib import Path
import uuid
import numpy as np

from .types import SAM3Error


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy() if value.is_floating_point() else value.detach().cpu().numpy()
    return np.asarray(value).copy()


def encode_mask(value):
    from pycocotools import mask as codec
    mask = np.asarray(value, dtype=np.uint8)
    if mask.ndim != 2:
        raise SAM3Error("invalid_mask", "Expected a two-dimensional binary mask")
    rle = codec.encode(np.asfortranarray(mask))
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


def decode_mask(value, height, width):
    from pycocotools import mask as codec
    if isinstance(value, dict):
        value = codec.decode(value)
    mask = np.asarray(value)
    if mask.shape != (height, width) or not np.isin(mask, [0, 1]).all():
        raise SAM3Error("invalid_mask", "Expected a binary mask at original frame resolution")
    return np.array(mask, dtype=bool, copy=True)


def mask_box(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else [0, 0, 0, 0]


def item(mask, score=None, **metadata):
    result = {"mask": encode_mask(mask), "bbox_xyxy": mask_box(mask), **metadata}
    if score is not None:
        result["score"] = float(score)
    return result


class ResultStore:
    """Append CPU frames to disk; no unbounded list of full-resolution masks."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifests = {}

    def begin(self, session_id, version):
        key = uuid.uuid4().hex
        (self.root / key).mkdir()
        self.manifests[key] = {"result_id": key, "session_id": session_id, "version": version,
                               "status": "running", "frames": [], "passes": []}
        self.finish(key, "running")
        return key

    def write(self, key, frame, direction):
        manifest = self.manifests[key]
        index = frame["frame_index"]
        # A later directional pass wins only for its explicitly emitted frame.
        target = self.root / key / f"{index}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(frame), encoding="utf-8")
        temporary.replace(target)
        if index not in manifest["frames"]:
            manifest["frames"].append(index)
        manifest["passes"].append({"frame": index, "direction": direction})

    def finish(self, key, status):
        manifest = self.manifests[key]
        manifest["status"] = status
        target = self.root / key / "manifest.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(target)
        return {**manifest, "frames": sorted(manifest["frames"]), "passes": list(manifest["passes"])}

    def read(self, key, session_id, version, frames):
        manifest = self.manifests.get(key)
        if manifest is None or manifest["session_id"] != session_id:
            raise SAM3Error("unknown_result", "Result does not belong to this runtime/session")
        if manifest["version"] != version:
            raise SAM3Error("stale_result", "Result belongs to an earlier session state")
        if len(frames) > 128:
            raise SAM3Error("invalid_range", "Read at most 128 frames per request")
        return [{"frame_index": frame, "availability": "not_computed"} if frame not in manifest["frames"]
                else dict(json.loads((self.root / key / f"{frame}.json").read_text()), availability="complete") for frame in frames]
