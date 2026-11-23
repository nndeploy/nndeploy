"""Algorithm extension contract, independent of tensors and native DAG bindings."""
from abc import ABC, abstractmethod


class AlgorithmAdapter(ABC):
    """Declare components and compose native graphs; resources own their lifetime.

    Hooks must release partial allocations if they fail before attaching a manager
    to resources. Attached managers are closed by DiTResources on all exit paths.
    Implementations and component classes may be registered as lazy import paths.
    """
    required_components = ()
    declared_model_index = None
    component_architectures = {}

    def capabilities(self):
        from nndeploy.e_dit.config.capabilities import AlgorithmCapabilities
        return AlgorithmCapabilities()

    def component_specs(self, config=None):
        from nndeploy.e_dit.loader.contracts import ComponentSpec
        return tuple(ComponentSpec(name) for name in self.required_components)

    def describe(self):
        return {
            'capabilities': self.capabilities().to_dict(),
            'components': [spec.to_dict() for spec in self.component_specs()],
        }

    @abstractmethod
    def build_config(self, params):
        raise NotImplementedError

    @abstractmethod
    def build_graph(self, params, name=None):
        raise NotImplementedError

    def build_workflow(self, params, **kwargs):
        raise NotImplementedError("This algorithm does not provide an editor workflow")

    def server_config(self, **kwargs):
        raise NotImplementedError("This algorithm does not provide a server schema")

    def config_scope(self, config):
        from nndeploy.e_dit.config.server_args import config_scope
        return config_scope(config)

    def validate_config(self, config):
        pass

    def validate_request(self, sampling, config):
        pass

    def validate_loading(self, config):
        pass

    def prepare_modules(self, modules, config):
        pass

    def compile_modules(self, modules, config):
        raise NotImplementedError("This algorithm does not declare component compilation")

    def initialize_resources(self, resources, config):
        pass
