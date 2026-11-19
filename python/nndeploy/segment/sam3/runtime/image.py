"""Serialized image runtime shared by the concept and interactive Graphs."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import threading
import uuid

import numpy as np

from ..modules import Modules
from ..results import array, item
from ..types import SAM3Error, box_xyxy, concept_boxes, fields, integer, points_xy


@dataclass
class ImageContext:
    state: dict
    height: int
    width: int
    version: int = 0
    threshold: float = 0.5
    result: dict = field(default_factory=dict)
    logits: dict = field(default_factory=dict)


class ImageRuntime:
    def __init__(self, config, modules=None):
        self.config = config
        self.modules = modules or Modules()
        self.lock = threading.RLock()
        self.contexts = {}
        self.model = self.processor = None
        self.closed = False

    @contextmanager
    def inference(self):
        import torch
        device = torch.device(self.config.device)
        with self.modules.activate(), torch.cuda.device(device) if device.type == "cuda" else nullcontext():
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=self.config.precision == "bfloat16"):
                yield

    def init(self):
        with self.lock:
            if self.closed:
                raise SAM3Error("runtime_closed", "Runtime has been closed")
            if self.model is None:
                from ..model_builder import build_sam3_image_model
                from ..model.sam3_image_processor import Sam3Processor
                import torch
                device = torch.device(self.config.device)
                with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
                    self.model = build_sam3_image_model(checkpoint_path=self.config.checkpoint,
                        device=device.type, load_from_HF=False, enable_inst_interactivity=True, compile=False)
                    self.processor = Sam3Processor(self.model, device=str(device))

    def close(self):
        with self.lock:
            self.contexts.clear()
            self.processor = self.model = None
            self.closed = True

    def _context(self, key):
        if key not in self.contexts:
            raise SAM3Error("unknown_context", "Image context is absent or released")
        return self.contexts[key]

    def _create(self, source):
        from PIL import Image, ImageOps
        if len(self.contexts) >= self.config.max_contexts:
            raise SAM3Error("resource_limit", "Release an image context before creating another")
        if isinstance(source, (str, bytes)):
            with Image.open(source) as loaded:
                image = ImageOps.exif_transpose(loaded).convert("RGB")
        elif isinstance(source, Image.Image):
            image = ImageOps.exif_transpose(source).convert("RGB")
        else:
            value = np.asarray(source)
            if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
                raise SAM3Error("invalid_image", "Array input must be HWC RGB uint8")
            image = Image.fromarray(value)
        state = self.processor.set_image(image)
        key = uuid.uuid4().hex
        self.contexts[key] = ImageContext(state, image.height, image.width)
        return key

    def execute(self, request, mode):
        op = request.get("op", mode)
        allowed = {"create": {"image"}, "query": {"context_id"}, "release": {"context_id"},
                   "clear": {"context_id"}, "concept": {"image", "context_id", "text", "boxes", "update", "threshold", "include_raw"},
                   "interact": {"image", "context_id", "groups", "include_raw"}}
        if op not in allowed or (op in ("concept", "interact") and op != mode):
            raise SAM3Error("invalid_operation", f"Operation {op} is not valid for this Graph")
        fields(request, allowed[op] | {"op"})
        if ("image" in request) == ("context_id" in request) and op != "create":
            raise SAM3Error("invalid_request", "Provide exactly one image or context_id")
        if op == "create" and "image" not in request:
            raise SAM3Error("invalid_request", "create requires image")
        with self.lock:
            self.init()
            with self.inference():
                temporary = "image" in request and op != "create"
                key = self._create(request["image"]) if "image" in request else request["context_id"]
                context = self._context(key)
                if "image" not in request and self.modules.observe:
                    self.modules.calls["image_context_hit"] += 1
                try:
                    if op == "release":
                        del self.contexts[key]
                        return {"context_id": key, "released": True}
                    if op == "clear":
                        self.processor.reset_all_prompts(context.state)
                        context.version += 1
                        context.result.clear(); context.logits.clear()
                    if op in ("create", "query", "clear"):
                        return {"context_id": key, "version": context.version, "size": [context.height, context.width]}
                    result = self._concept(context, request) if op == "concept" else self._interact(context, request)
                    result.update(context_id=None if temporary else key, version=context.version, size=[context.height, context.width])
                    # Keep only private selection metadata. Public results and
                    # optional raw arrays may be freely mutated by their caller.
                    context.result = {"result_id": result["result_id"], "kind": result["kind"]}
                    if result["kind"] == "concept":
                        context.result["instances"] = [{"model_bbox_xyxy": list(x["model_bbox_xyxy"])} for x in result["instances"]]
                    if temporary:
                        result["result_id"] = None
                    return result
                finally:
                    if temporary:
                        self.contexts.pop(key, None)

    def _concept(self, context, request):
        update = request.get("update", "replace")
        if update not in ("replace", "append"):
            raise SAM3Error("invalid_request", "update must be replace or append")
        text = request.get("text")
        if text is not None and (not isinstance(text, str) or not text.strip()):
            raise SAM3Error("invalid_request", "text must be nonempty")
        boxes, labels = concept_boxes(request.get("boxes", []), context.width, context.height)
        if update == "append" and text is not None:
            raise SAM3Error("invalid_request", "append accepts boxes; replace is required to change text")
        if update == "replace" and text is None and not boxes:
            raise SAM3Error("invalid_request", "A concept query requires text or boxes")
        if update == "replace" and text is None and not any(labels):
            raise SAM3Error("unsupported_prompt", "A visual-only query needs a positive box")
        threshold = request.get("threshold", context.threshold)
        if not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1:
            raise SAM3Error("invalid_request", "threshold must be between 0 and 1")
        if update == "append" and not boxes and "boxes" not in context.state:
            raise SAM3Error("invalid_request", "There is no previous concept result to update")
        context.version += 1
        context.result.clear(); context.logits.clear()
        context.threshold = float(threshold)
        self.processor.confidence_threshold = context.threshold
        try:
            if update == "replace":
                self.processor.reset_all_prompts(context.state)
            if text is not None:
                self.processor.set_text_prompt(text, context.state)
            for box, label in zip(boxes, labels):
                self.processor.add_geometric_prompt(box, bool(label), context.state)
            if not boxes and text is None:
                self.processor.set_confidence_threshold(context.threshold, context.state)
            state = context.state
            masks, scores, bounds = array(state["masks"])[:, 0], array(state["scores"]), array(state["boxes"])
            result = {"result_id": uuid.uuid4().hex, "kind": "concept", "instances": [
                item(mask, score, index=i, score_kind="concept", model_bbox_xyxy=bounds[i].tolist())
                for i, (mask, score) in enumerate(zip(masks, scores))]}
            if request.get("include_raw", False):
                result["raw"] = {"masks": masks, "scores": scores, "boxes": bounds, "probabilities": array(state["masks_logits"])[:, 0]}
            return result
        except Exception:
            self.processor.reset_all_prompts(context.state)
            raise

    def _interact(self, context, request):
        groups = request.get("groups")
        if not isinstance(groups, list) or not groups:
            raise SAM3Error("invalid_request", "interact requires nonempty groups")
        prepared, identifiers = [], set()
        for index, group in enumerate(groups):
            fields(group, {"id", "points", "labels", "box", "selection", "refine", "mask_logits", "multimask"})
            identity = group.get("id", str(index))
            if not isinstance(identity, str) or identity in identifiers:
                raise SAM3Error("invalid_request", "Group ids must be unique strings")
            identifiers.add(identity)
            params = {}
            if "points" in group:
                params["point_coords"] = np.asarray(points_xy(group["points"], group.get("labels", []), context.width, context.height), np.float32)
                params["point_labels"] = np.asarray(group["labels"], np.int32)
            elif "labels" in group:
                raise SAM3Error("invalid_points", "labels requires points")
            if "box" in group:
                params["box"] = np.asarray(box_xyxy(group["box"], context.width, context.height), np.float32)
            if "selection" in group:
                if "box" in group:
                    raise SAM3Error("invalid_request", "selection and box are mutually exclusive")
                selection = group["selection"]
                fields(selection, {"result_id", "index"}, {"result_id", "index"})
                prior = context.result
                if prior.get("result_id") != selection["result_id"] or prior.get("kind") != "concept":
                    raise SAM3Error("stale_result", "Selection requires the current concept result")
                selected = integer(selection["index"], "index")
                if selected >= len(prior["instances"]):
                    raise SAM3Error("invalid_request", "Selected instance does not exist")
                params["box"] = np.asarray(prior["instances"][selected]["model_bbox_xyxy"], np.float32)
            if "refine" in group:
                if "mask_logits" in group:
                    raise SAM3Error("invalid_request", "Use either refine or mask_logits")
                reference = group["refine"]
                fields(reference, {"result_id", "group_id", "candidate"}, {"result_id", "group_id", "candidate"})
                if context.result.get("result_id") != reference["result_id"]:
                    raise SAM3Error("stale_result", "Refinement requires the current interactive result")
                prior = context.logits.get(reference["group_id"])
                candidate = integer(reference["candidate"], "candidate")
                if prior is None or candidate >= len(prior):
                    raise SAM3Error("invalid_request", "Refinement candidate does not exist")
                params["mask_input"] = prior[candidate][None].copy()
            elif "mask_logits" in group:
                logits = np.asarray(group["mask_logits"], np.float32)
                expected = tuple(self.model.inst_interactive_predictor.model.sam_prompt_encoder.mask_input_size)
                if logits.shape != (1, *expected) or not np.isfinite(logits).all():
                    raise SAM3Error("invalid_logits", f"mask_logits requires finite shape {(1, *expected)}")
                params["mask_input"] = logits
            if not params:
                raise SAM3Error("invalid_request", "Each group requires instance prompts")
            multimask = group.get("multimask", True)
            if not isinstance(multimask, bool):
                raise SAM3Error("invalid_request", "multimask must be boolean")
            params["multimask_output"] = multimask
            prepared.append((identity, params))
        context.version += 1
        context.result.clear(); context.logits.clear()
        result = {"result_id": uuid.uuid4().hex, "kind": "interactive", "groups": []}
        predictor = self.model.inst_interactive_predictor
        try:
            for identity, params in prepared:
                try:
                    masks, scores, logits = self.model.predict_inst(context.state, **params)
                finally:
                    predictor._features = None
                    predictor._is_image_set = False
                masks, scores, logits = array(masks), array(scores), array(logits)
                context.logits[identity] = logits.copy()
                entry = {"id": identity, "candidates": [item(mask, score, index=i, score_kind="mask_quality") for i, (mask, score) in enumerate(zip(masks, scores))], "logits_shape": list(logits.shape[1:])}
                if request.get("include_raw", False):
                    entry["raw"] = {"masks": masks, "scores": scores, "logits": logits}
                result["groups"].append(entry)
            return result
        except Exception:
            context.logits.clear()
            raise
