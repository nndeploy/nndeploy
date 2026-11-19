"""Three high-level Graphs composed from preparation, inference and result Nodes."""

import nndeploy.dag
from .nodes import (Creator, PREFIX, SAM3InputNode, SAM3ResultNode,
                    SAM3ImageConceptNode, SAM3ImageInteractiveNode, SAM3VideoNode)


class _SAM3Graph(nndeploy.dag.Graph):
    inference_type = None

    def __init__(self, name, inputs=None, outputs=None, owner=None):
        inputs = [nndeploy.dag.Edge("request")] if not inputs else inputs
        outputs = [nndeploy.dag.Edge("result")] if not outputs else outputs
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("SAM3 Graphs require one request and one result edge")
        super().__init__(name, inputs, outputs)
        self.set_key(PREFIX + type(self).__name__)
        self.set_input_type(dict)
        self.set_output_type(dict)
        prepared = nndeploy.dag.Edge(name + "/prepared")
        inferred = nndeploy.dag.Edge(name + "/inferred")
        for edge in (prepared, inferred):
            self.add_edge(edge)
        for node in (SAM3InputNode(name + "/prepare", inputs, [prepared]),
                     self.inference_type(name + "/infer", [prepared], [inferred]),
                     SAM3ResultNode(name + "/result", [inferred], outputs)):
            self.add_node(node)
            if node not in self.nodes:
                self.nodes.append(node)
        if owner is not None:
            from .workflow import bind_runtime
            bind_runtime(self, owner)


class SAM3ImageConceptGraph(_SAM3Graph):
    inference_type = SAM3ImageConceptNode


class SAM3ImageInteractiveGraph(_SAM3Graph):
    inference_type = SAM3ImageInteractiveNode


class SAM3VideoGraph(_SAM3Graph):
    inference_type = SAM3VideoNode


_creators = []
for _cls in (SAM3ImageConceptGraph, SAM3ImageInteractiveGraph, SAM3VideoGraph):
    _creator = Creator(_cls)
    _creators.append(_creator)
    nndeploy.dag.register_node(PREFIX + _cls.__name__, _creator)
