"""Native state graphs and dynamic loops; no separate model executor."""
from dataclasses import dataclass
from contextvars import ContextVar

loading_graph = ContextVar("dit_loading_graph", default=False)
from nndeploy.base import Status, StatusCode
from nndeploy.dag import Edge, Graph, Loop, NodeCreator, register_node


@dataclass
class StateSlot:
    state: object = None
    error: BaseException | None = None


_CREATORS = []
_NODE_TYPES = {}

def register_type(cls):
    _NODE_TYPES["nndeploy.e_dit." + cls.__name__] = cls
    class Creator(NodeCreator):
        def create_node(self, name, inputs, outputs):
            token = loading_graph.set(True)
            try:
                node = cls(name=name, inputs=inputs, outputs=outputs)
                self.node = node
                return node
            finally:
                loading_graph.reset(token)
    creator = Creator()
    _CREATORS.append(creator)
    register_node('nndeploy.e_dit.' + cls.__name__, creator)
    return cls


def materialize_children(graph, data):
    """Create Python-owned nodes from the saved topology before native linking."""
    if graph.get_node_count() == 0:
        children = []
        token = loading_graph.set(True)
        try:
            for item in data.get('node_repository_', []):
                cls = _NODE_TYPES.get(item['key_'])
                if cls is None:
                    raise ValueError(f"Unsupported model graph node: {item['key_']}")
                child = cls(name=item['name_'], inputs=[], outputs=[])
                graph.add_node(child)
                children.append(child)
            graph._children = children
        finally:
            loading_graph.reset(token)
    for child, item in zip(graph._children, data.get('node_repository_', [])):
        if 'node_repository_' in item:
            materialize_children(child, item)


def release_bindings(node):
    """Rollback nodes configured before native graph initialization completed."""
    error = None
    for child in getattr(node, '_children', []):
        try:
            release_bindings(child)
        except BaseException as exc:
            if error is None: error = exc
    try:
        close = getattr(node, 'close', None)
        if callable(close): close()
    except BaseException as exc:
        if error is None: error = exc
    finally:
        release = getattr(node, 'release_bindings', None)
        if callable(release): release()
        if hasattr(node, '_resources'): node._resources = None
    if error is not None: raise error


def clear_ports(node):
    for child in getattr(node, '_children', []):
        clear_ports(child)
    node.set_inputs([])
    node.set_outputs([])


class StateGraph(Graph):
    def __init__(self, name='state_graph', inputs=None, outputs=None):
        super().__init__(name, [Edge(name+'_in')] if inputs is None else inputs, [Edge(name+'_out')] if outputs is None else outputs)
        self.set_key('nndeploy.e_dit.' + type(self).__name__)
        self.set_input_type(object)
        self.set_output_type(object)
        self._children = []
        self._ready = False
        self._resources = None
        self._slot = StateSlot()

    def set_inputs(self, inputs):
        status = super().set_inputs(inputs)
        if self._children:
            self._children[0].set_inputs(inputs)
        return status

    def set_outputs(self, outputs):
        status = super().set_outputs(outputs)
        if self._children:
            self._children[-1].set_outputs(outputs)
        return status

    def compose(self, nodes):
        if loading_graph.get(): return
        self._children = list(nodes)
        self._edges = [self.get_input(0), self.get_output(0)]
        previous = self.get_input(0)
        for i, node in enumerate(self._children):
            output = self.get_output(0) if i == len(self._children)-1 else Edge(self.get_name()+f'_{i}')
            self._edges.append(output)
            self.add_edge(output)
            node.set_inputs([previous])
            node.set_outputs([output])
            self.add_node(node)
            previous = output

    def bind(self, resources):
        self._children = [self.get_node(i) for i in range(self.get_node_count())]
        self._resources = resources
        for child in self._children:
            child.bind(resources)

    def init(self):
        status = super().init()
        self._ready = status == StatusCode.Ok
        return status

    def invoke(self, state):
        if not self._ready:
            status = self.init()
            if status != StatusCode.Ok: raise RuntimeError(f'Graph initialization failed: {status}')
            self._ready = True
        slot = self._slot
        slot.state, slot.error = state, None
        try:
            self.get_input(0).set(slot)
            status = self.run()
            if slot.error is not None: raise slot.error
            if status != StatusCode.Ok: raise RuntimeError(f'Graph execution failed: {status}')
            return slot.state
        finally:
            slot.state = slot.error = None

    def deinit(self):
        try:
            return super().deinit() if self._ready else Status.ok()
        finally:
            self._ready = False
            self._resources = None
            self._slot.state = self._slot.error = None

    def deserialize(self, value):
        for child in self._children:
            clear_ports(child)
        status = super().deserialize(value)
        self._children = [self.get_node(i) for i in range(self.get_node_count())]
        return status


class IteratorLoop(Loop):
    """Drive a lazily planned sequence with the native Loop executor.

    A generator supplies each iteration's State and receives its result. Native
    Loop schedules the registered body nodes; generator.close unwinds resources
    on cancellation, failed nodes, and partially consumed plans.
    """
    def __init__(self, name='iterations', inputs=None, outputs=None):
        input_edges = [Edge(name+'_in')] if inputs is None else inputs
        output_edges = [Edge(name+'_out')] if outputs is None else outputs
        super().__init__(name, input_edges, output_edges)
        self.set_key('nndeploy.e_dit.' + type(self).__name__)
        self.set_input_type(object)
        self.set_output_type(object)
        self._children = []
        self.nodes = []
        self._ports = input_edges + output_edges
        for edge in self._ports:
            self.add_edge(edge)
        self._iterator = None
        self._index = 0
        self._slot = StateSlot()

    def set_inputs(self, inputs):
        status = super().set_inputs(inputs)
        if self._children:
            self._children[0].set_inputs(inputs)
        return status

    def set_outputs(self, outputs):
        status = super().set_outputs(outputs)
        if self._children:
            self._children[-1].set_outputs(outputs)
        return status

    compose = StateGraph.compose
    bind = StateGraph.bind

    def deserialize(self, value):
        from nndeploy import _nndeploy_internal as _C
        for child in self._children:
            clear_ports(child)
        status = _C.dag.Graph.deserialize(self, value)
        self._children = [self.get_node(i) for i in range(self.get_node_count())]
        return status

    def deinit(self):
        try:
            return super().deinit()
        finally:
            self._resources = None
            self._slot.state = self._slot.error = None

    def iterations(self, state):
        raise NotImplementedError

    def loops(self):
        try:
            value = next(self._iterator) if self._index == 0 else self._iterator.send(self._slot.state)
        except StopIteration:
            return self._index
        except BaseException as error:
            self._slot.error = error
            return self._index
        self._slot.state = value
        self.get_input(0).set(self._slot)
        self._index += 1
        return self._index

    def run(self):
        outer = self.get_input(0).get(self)
        original = outer.state
        self._slot.state, self._slot.error = None, None
        self._index = 0
        self._iterator = self.iterations(original)
        status = Status.ok()
        try:
            status = super().run()
            if self._slot.error is not None: raise self._slot.error
        except BaseException as error:
            outer.error = error
            status = Status(StatusCode.ErrorInvalidValue)
        finally:
            try:
                self._iterator.close()
            except BaseException as error:
                if outer.error is None: outer.error = error
                status = Status(StatusCode.ErrorInvalidValue)
            finally:
                self._iterator = None
                self._slot.state = self._slot.error = None
                outer.state = original
                self.get_input(0).set(outer)
                self.get_output(0).set(outer)
        return status
