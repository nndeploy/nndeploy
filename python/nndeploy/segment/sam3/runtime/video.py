"""Single-device, owner-serialized video sessions with cooperative cancellation."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import threading

from ..modules import Modules, PropagationCancelled, check_cancel
from ..results import ResultStore, array, decode_mask, item
from ..types import SAM3Error, concept_boxes, fields, integer, points_xy


@dataclass
class VideoSession:
    state: dict
    version: int = 0
    status: str = "ready"
    cancel: threading.Event = field(default_factory=threading.Event)
    active: bool = False
    result_id: str = None
    prompted: bool = False


class VideoRuntime:
    OPERATIONS = {
        "start": {"source"}, "query": {"session_id"}, "close": {"session_id"},
        "reset": {"session_id"}, "cancel": {"session_id"},
        "concept": {"session_id", "frame", "text", "boxes", "restart"},
        "points": {"session_id", "frame", "object_id", "points", "labels"},
        "mask": {"session_id", "frame", "object_id", "mask"},
        "remove": {"session_id", "object_id"},
        "propagate": {"session_id", "start", "end", "direction"},
        "read": {"session_id", "result_id", "frames"},
    }

    def __init__(self, config, result_dir, modules=None):
        self.config = config
        self.modules = modules or Modules()
        self.store = ResultStore(result_dir)
        self.lock = threading.RLock()
        self.control = threading.Lock()
        self.sessions = {}
        self.predictor = None
        self.closed = False

    @contextmanager
    def inference(self):
        import torch
        with self.modules.activate(), torch.cuda.device(self.config.device), torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            yield

    def init(self):
        with self.lock:
            if self.closed:
                raise SAM3Error("runtime_closed", "Runtime has been closed")
            if self.predictor is None:
                if not self.config.device.startswith("cuda") or self.config.precision != "bfloat16":
                    raise SAM3Error("unsupported_config", "Native video currently requires CUDA and bfloat16")
                import torch
                from ..model_builder import build_sam3_video_predictor
                index = torch.device(self.config.device).index
                self.predictor = build_sam3_video_predictor(checkpoint_path=self.config.checkpoint,
                    gpus_to_use=[index if index is not None else torch.cuda.current_device()], compile=False)

    def close(self):
        with self.lock:
            if self.predictor is not None:
                for key in list(self.sessions):
                    self.predictor.close_session(key)
                self.predictor.shutdown()
            with self.control:
                self.sessions.clear()
            self.predictor = None
            self.closed = True

    def cancel(self, session_id):
        # This control path never waits for the long-running inference lock.
        with self.control:
            session = self.sessions.get(session_id)
            if session is None:
                raise SAM3Error("unknown_session", "Session is absent or closed")
            if session.active:
                session.cancel.set()
                return {"session_id": session_id, "status": "cancellation_requested"}
            return {"session_id": session_id, "status": session.status, "active": False}

    def _get(self, key):
        if key not in self.sessions:
            raise SAM3Error("unknown_session", "Session is absent or closed")
        return self.sessions[key]

    def _info(self, key, session):
        state = session.state
        ids = state.get("tracker_metadata", {}).get("obj_ids_all_gpu", [])
        return {"session_id": key, "version": session.version, "status": session.status,
                "size": [state["orig_height"], state["orig_width"]], "num_frames": state["num_frames"],
                "object_ids": [int(x) for x in ids], "result_id": session.result_id}

    def _frame(self, response, session):
        data = response["outputs"]
        ids, masks = array(data["out_obj_ids"]), array(data["out_binary_masks"])
        scores = array(data["out_probs"]) if "out_probs" in data else None
        return {"frame_index": int(response["frame_index"]), "version": session.version,
                "size": [session.state["orig_height"], session.state["orig_width"]],
                "objects": [item(mask, scores[i] if scores is not None else None,
                    object_id=int(identity), score_kind="object_score") for i, (identity, mask) in enumerate(zip(ids, masks))]}

    def execute(self, request):
        op = request.get("op")
        if op not in self.OPERATIONS:
            raise SAM3Error("invalid_operation", "Unknown video operation")
        fields(request, self.OPERATIONS[op] | {"op"}, {"op", "source" if op == "start" else "session_id"})
        if op == "cancel":
            return self.cancel(request["session_id"])
        with self.lock:
            self.init()
            with self.inference():
                if op == "start":
                    if len(self.sessions) >= self.config.max_sessions:
                        raise SAM3Error("resource_limit", "Close a video session before opening another")
                    response = self.predictor.start_session(resource_path=request["source"])
                    key = response["session_id"]
                    state = self.predictor._get_session(key)["state"]
                    session = VideoSession(state)
                    state["_sam3_workflow_cancel"] = session.cancel
                    with self.control:
                        self.sessions[key] = session
                    return self._info(key, session)
                key = request["session_id"]
                session = self._get(key)
                if op == "query":
                    return self._info(key, session)
                if op == "close":
                    self.predictor.close_session(key)
                    with self.control:
                        del self.sessions[key]
                    return {"session_id": key, "closed": True}
                if op == "read":
                    frames = request.get("frames", [])
                    if not isinstance(frames, list):
                        raise SAM3Error("invalid_request", "frames must be an integer list")
                    for index in frames:
                        self._index(index, session)
                    return {"session_id": key, "frames": self.store.read(request.get("result_id"), key, session.version, frames)}
                if op == "reset":
                    self.predictor.reset_session(key)
                    session.version += 1
                    session.cancel.clear()
                    session.state["_sam3_workflow_cancel"] = session.cancel
                    session.status, session.result_id = "ready", None
                    session.prompted = False
                    return self._info(key, session)
                if session.status in ("cancelled", "failed"):
                    raise SAM3Error("reset_required", "Reset or close this session before further inference")
                if op == "propagate":
                    return self._propagate(key, session, request)
                return self._prompt(key, session, op, request)

    def _index(self, index, session):
        index = integer(index, "frame")
        if index >= session.state["num_frames"]:
            raise SAM3Error("invalid_frame", "Frame is outside the video")
        return index

    def _prompt(self, key, session, op, request):
        height, width = session.state["orig_height"], session.state["orig_width"]
        response = {"type": "add_prompt", "session_id": key}
        if op != "remove":
            response["frame_index"] = self._index(request.get("frame"), session)
        if op == "concept":
            if request.get("restart") is not True:
                raise SAM3Error("restart_required", "Concept changes reset tracking; explicitly pass restart=true")
            text = request.get("text")
            boxes, labels = concept_boxes(request.get("boxes", []), width, height, video=True)
            if text is not None and (not isinstance(text, str) or not text.strip()):
                raise SAM3Error("invalid_request", "text must be nonempty")
            if len(boxes) > 1:
                raise SAM3Error("unsupported_prompt", "Native video accepts one initial concept box")
            if text is None and boxes and not any(labels):
                raise SAM3Error("unsupported_prompt", "A visual-only query needs a positive box")
            if text is None and not boxes:
                raise SAM3Error("invalid_request", "Concept requires text or a concept box")
            if text is not None:
                response["text"] = text
            if boxes:
                response.update(bounding_boxes=boxes, bounding_box_labels=labels)
        elif op in ("points", "mask", "remove"):
            identity = integer(request.get("object_id"), "object_id")
            response["obj_id"] = identity
            if op == "points":
                points = points_xy(request.get("points", []), request.get("labels", []), width, height)
                response.update(points=points, point_labels=request["labels"], rel_coordinates=False)
            elif op == "mask":
                response.update(type="add_mask", mask=decode_mask(request.get("mask"), height, width))
            else:
                if identity not in self._info(key, session)["object_ids"]:
                    raise SAM3Error("unknown_object", "Object does not exist in this session")
                response.update(type="remove_object", is_user_action=False)
        session.version += 1
        session.result_id = None
        try:
            result = self.predictor.handle_request(response)
            if op != "remove":
                session.prompted = True
            session.state["_sam3_workflow_cancel"] = session.cancel
            return {**self._info(key, session), **({"frame": self._frame(result, session)} if "outputs" in result else {})}
        except Exception:
            session.status = "failed"
            raise

    def _propagate(self, key, session, request):
        if not session.prompted:
            raise SAM3Error("prompt_required", "Add concept or instance prompts before propagation")
        start = integer(request.get("start", 0), "start")
        end = integer(request.get("end", session.state["num_frames"]), "end", 1)
        direction = request.get("direction", "forward")
        if not (start < end <= session.state["num_frames"]) or direction not in ("forward", "backward", "both"):
            raise SAM3Error("invalid_range", "Expected [start,end) inside the video and a valid direction")
        result_id = self.store.begin(key, session.version)
        session.result_id = result_id
        with self.control:
            session.cancel.clear()
            session.active = True
            session.status = "running"
        state = session.state
        state["_sam3_workflow_cancel"] = session.cancel
        state["_sam3_workflow_include_reverse_start"] = True
        try:
            directions = ["forward", "backward"] if direction == "both" else [direction]
            for current in directions:
                check_cancel(state)
                stream = self.predictor.handle_stream_request({"type": "propagate_in_video", "session_id": key,
                    "propagation_direction": current, "start_frame_index": start if current == "forward" else end - 1,
                    "max_frame_num_to_track": end - start - 1,
                    "force_tracker_propagation": direction == "both" and current == "backward"})
                try:
                    for response in stream:
                        check_cancel(state)
                        if start <= response["frame_index"] < end:
                            self.store.write(result_id, self._frame(response, session), current)
                finally:
                    stream.close()
            with self.control:
                check_cancel(state)
                session.status = "ready"
                session.active = False
            manifest = self.store.finish(result_id, "succeeded")
        except PropagationCancelled:
            session.status = "cancelled"
            manifest = self.store.finish(result_id, "cancelled")
        except Exception:
            session.status = "failed"
            self.store.finish(result_id, "failed")
            raise
        finally:
            state.pop("_sam3_workflow_include_reverse_start", None)
            with self.control:
                session.active = False
        return {**self._info(key, session), "result": manifest}
