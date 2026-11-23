"""Shared model ownership; request execution belongs to native graphs."""
from __future__ import annotations
import os
from nndeploy.e_dit.utils.env import get_env
import time
from nndeploy.e_dit.models.components import ComponentOwner
from nndeploy.e_dit.config.server_args import set_global_server_args
from nndeploy.e_dit.utils.logging_utils import init_logger
from nndeploy.e_dit.utils.cpu_resources import configure_process_cpu_resources
from nndeploy.e_dit.utils.determinism import enable_deterministic_mode
from nndeploy.e_dit.parallel.runtime import initialize_runtime_distributed
logger = init_logger(__name__)

class DiTResources(ComponentOwner):
    """Shared weights, configuration and distributed resources for native nodes."""
    def __init__(self, params):
        from threading import Lock
        from nndeploy.e_dit.registry import get_model
        self.params = params
        self.adapter = get_model(params.model)
        self.config = None
        self.modules = {}
        self._extra_config_module_map = {}
        self.memory_usages = {}
        self.loading_report = []
        self._owns_distributed = False
        self._managers = []
        self._closed = True
        self._cpu_resources = None
        self._warmed_up = False
        self._lock = Lock()

    def open(self):
        if not self._closed:
            return self
        self.config = self.params.build_config()
        self.server_args = server_args = self.config
        self.model_path = server_args.model_path
        self._closed = False
        with self.adapter.config_scope(server_args):
            try:
                self._open_components(server_args)
            except BaseException:
                self.close()
                raise
        return self

    def _open_components(self, server_args):
        # Must run before any CUDA work so the cuBLAS workspace config and the
        # deterministic kernels are in place for the whole session.
        enable_deterministic_mode()
        initialize_start = time.perf_counter()
        local_world_size = int(
            os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1"))
        )
        if get_env("NNDEPLOY_DIT_CPU_AFFINITY_MAP") or local_world_size > 1:
            self._cpu_resources = configure_process_cpu_resources()
        distributed_context = getattr(server_args, "distributed_context", None)
        if distributed_context is None:
            distributed_context = initialize_runtime_distributed(server_args)
            self._owns_distributed = True
        self.distributed_context = distributed_context
        from nndeploy.e_dit.memory.allocator import configure_cuda_allocator
        configure_cuda_allocator(server_args.cuda_memory_limit_gib, server_args.device)
        set_global_server_args(server_args)
        self.distributed_setup_s = time.perf_counter() - initialize_start
        self.modules = self.load_modules(server_args)
        self.initialize_resources(server_args)
        self._component_compile = None
        if server_args.compile_components:
            self._component_compile = self.adapter.compile_modules(self.modules, server_args)

    @property
    def components(self):
        return self.modules

    def register_manager(self, manager):
        """Transfer cleanup ownership for an algorithm resource to this owner."""
        self._managers.append(manager)
        return manager

    def close(self):
        if self._closed:
            return
        error = None
        managers, self._managers = self._managers, []
        callbacks = [manager.close for manager in reversed(managers)]
        compile_manager = getattr(self, '_component_compile', None)
        memory = getattr(self, '_memory_adapter', None)
        if compile_manager:
            callbacks.append(compile_manager.close)
        if memory:
            callbacks.append(lambda: memory.shutdown(terminal=True))
        if self._owns_distributed:
            from nndeploy.e_dit.parallel.runtime import destroy_runtime_distributed, get_runtime_distributed_context
            if get_runtime_distributed_context() is getattr(self, 'distributed_context', None):
                callbacks.append(destroy_runtime_distributed)
        try:
            for close in callbacks:
                try:
                    close()
                except BaseException as exc:
                    if error is None:
                        error = exc
        finally:
            self.modules.clear()
            self._memory_adapter = None
            self._component_compile = None
            self._closed = True
            self._warmed_up = False
            self._owns_distributed = False
        if error is not None:
            raise error

    @property
    def _required_config_modules(self):
        return self.adapter.required_components

    @property
    def _declared_model_index(self):
        return self.adapter.declared_model_index

    def load_modules(self, server_args, loaded_modules=None):
        self.adapter.validate_loading(server_args)
        from nndeploy.e_dit.memory.telemetry import memory_observation
        self._initialization_memory = {"before_loading": memory_observation(server_args.device)}
        modules = super().load_modules(server_args, loaded_modules)
        self.modules = modules  # Own allocations before preparation can fail.
        self.adapter.prepare_modules(modules, server_args)
        self._initialization_memory["after_loading"] = memory_observation(server_args.device)
        return modules

    def initialize_resources(self, server_args):
        self.adapter.initialize_resources(self, server_args)
