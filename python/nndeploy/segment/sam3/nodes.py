"""Workflow Nodes. Import explicitly to register SAM3 without public changes."""

import json
import time
import nndeploy.base
import nndeploy.dag

from .types import SAM3Error

PREFIX = "nndeploy.segment.sam3."


class SAM3InputNode(nndeploy.dag.Node):
    def __init__(self, name, inputs=None, outputs=None):
        super().__init__(name, inputs, outputs)
        self.set_key(PREFIX + type(self).__name__)
        self.set_input_type(dict)
        self.set_output_type(dict)

    def run(self):
        request = self.get_input(0).get(self)
        if not isinstance(request, dict):
            raise SAM3Error("invalid_request", "Graph input must be a request object")
        self.get_output(0).set(dict(request))
        return nndeploy.base.Status.ok()


class SAM3ResultNode(SAM3InputNode):
    """Pass an independently owned CPU result to downstream workflow components."""


class _InferenceNode(SAM3InputNode):
    mode = None

    def __init__(self, name, inputs=None, outputs=None):
        super().__init__(name, inputs, outputs)
        self.owner = None
        self.model_config = {}

    def init(self):
        if self.owner is None:
            raise SAM3Error("owner_required", "Bind a RuntimeOwner with sam3.workflow.bind_runtime before init")
        runtime = self.owner.video if self.mode == "video" else self.owner.image
        runtime.init()
        return nndeploy.base.Status.ok()

    def deinit(self):
        # Nodes borrow resources. Only RuntimeOwner.close destroys shared models.
        return nndeploy.base.Status.ok()

    def run(self):
        request = self.get_input(0).get(self)
        started = time.perf_counter()
        try:
            runtime = self.owner.video if self.mode == "video" else self.owner.image
            data = runtime.execute(request) if self.mode == "video" else runtime.execute(request, self.mode)
            response = {"ok": True, **data}
        except SAM3Error as error:
            response = {"ok": False, "error": {"code": error.code, "message": str(error)}}
        response["cpu_wall_seconds"] = time.perf_counter() - started
        self.get_output(0).set(response)
        return nndeploy.base.Status.ok()

    def serialize(self):
        data = json.loads(super().serialize())
        data["model_config"] = dict(self.model_config)
        return json.dumps(data)

    def deserialize(self, value):
        data = json.loads(value)
        self.model_config = data.get("model_config", {})
        return super().deserialize(value)


class SAM3ImageConceptNode(_InferenceNode):
    mode = "concept"


class SAM3ImageInteractiveNode(_InferenceNode):
    mode = "interact"


class SAM3VideoNode(_InferenceNode):
    mode = "video"


class Creator(nndeploy.dag.NodeCreator):
    def __init__(self, cls):
        super().__init__()
        self.cls = cls

    def create_node(self, name, inputs, outputs):
        node = self.cls(name, inputs, outputs)
        # The Python Graph wrapper retains created nodes. Retain the latest
        # here as the repository template does, without accumulating every run.
        self.node = node
        return node


_creators = []
for _cls in (SAM3InputNode, SAM3ResultNode, SAM3ImageConceptNode, SAM3ImageInteractiveNode, SAM3VideoNode):
    _creator = Creator(_cls)
    _creators.append(_creator)
    nndeploy.dag.register_node(PREFIX + _cls.__name__, _creator)
