"""Execute a numerical generator one step at a time through native DAG Loop."""
from .base import StateNode, operation
from .graph import IteratorLoop, register_type
from .control import service_checkpoint

@register_type
class GeneratorStepNode(StateNode):
    @operation
    def run(self, batch, args):
        service_checkpoint(batch, args, phase='denoise_step')
        owner = self._owner
        try:
            next(owner._steps)
        except StopIteration:
            owner._steps_done = True
        return batch

class GeneratorLoop(IteratorLoop, StateNode):
    def __init__(self, name=None, inputs=None, outputs=None, **bindings):
        IteratorLoop.__init__(self, name or type(self).__name__, inputs, outputs)
        self._bindings = bindings
        self._configured = False
        self._enable_logging = True
        self.compose([GeneratorStepNode('step')])

    def bind(self, resources):
        StateNode.bind(self, resources)
        IteratorLoop.bind(self, resources)
        for child in self._children:
            child._owner = self

    def iterations(self, batch):
        self._steps = self.compute(batch, self.server_args)
        self._steps_done = False
        try:
            try:
                next(self._steps)
            except StopIteration:
                return
            while not self._steps_done:
                yield batch
        finally:
            self._steps.close()
            self._steps = None
            self._active_batch = None
            self.scheduler = self.resources.modules["scheduler"]
            if self.server_args.dit_cpu_offload:
                self.transformer.to('cpu')

    def deinit(self):
        try:
            return IteratorLoop.deinit(self)
        finally:
            self.release_bindings()
