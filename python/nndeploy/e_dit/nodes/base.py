"""Native Node state binding, validation, metrics and distributed error handling."""

from __future__ import annotations

from functools import wraps
import inspect
from nndeploy.dag import Node
from nndeploy.base import Status, StatusCode
from nndeploy.e_dit.parallel.stage_policy import synchronize_stage_error

import torch

from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.nodes.validators import VerificationResult
from nndeploy.e_dit.parallel.stage_policy import StageExecutionPolicy
from nndeploy.e_dit.utils.logging_utils import init_logger
from nndeploy.e_dit.utils.perf_logger import StageProfiler
from nndeploy.e_dit.utils.platform import current_platform

logger = init_logger(__name__)


class StageVerificationError(Exception):
    """Raised when a stage input/output contract is violated."""


class StateNode(Node):
    """A native DAG node sharing request state and resident model resources."""

    def __init__(self, name=None, inputs=None, outputs=None, **bindings):
        super().__init__(name or type(self).__name__, inputs or [], outputs or [])
        self.set_key('nndeploy.e_dit.' + type(self).__name__)
        self.set_input_type(object)
        self.set_output_type(object)
        self.server_args = None
        self._bindings = bindings
        self._enable_logging = True
        self._configured = False
        self.resources = None

    def bind(self, resources):
        self.resources = resources
        self.server_args = resources.config
        if not self._configured:
            setup = getattr(self, 'setup', None)
            if setup is not None:
                available = {**resources.modules, 'server_args': resources.config, **self._bindings}
                before = set(self.__dict__)
                try:
                    setup(**{key: available[key] for key in inspect.signature(setup).parameters if key in available})
                finally:
                    self._setup_attributes = set(self.__dict__) - before
            self._configured = True

    def deinit(self):
        close = getattr(self, 'close', None)
        try:
            if close is not None:
                close()
        finally:
            self.release_bindings()
        return Status.ok()

    def release_bindings(self):
        for key in getattr(self, '_setup_attributes', ()):
            self.__dict__.pop(key, None)
        self._setup_attributes = set()
        self._configured = False
        self.resources = None
        self.server_args = None

    def log_info(self, msg: str, *args) -> None:
        if self._enable_logging:
            logger.info("[%s] " + msg, self.__class__.__name__, *args)

    def log_warning(self, msg: str, *args) -> None:
        logger.warning("[%s] " + msg, self.__class__.__name__, *args)

    def log_error(self, msg: str, *args) -> None:
        logger.error("[%s] " + msg, self.__class__.__name__, *args)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        return VerificationResult()

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        return VerificationResult()

    @property
    def execution_policy(self) -> StageExecutionPolicy:
        return StageExecutionPolicy.replicated()

    @property
    def device(self) -> torch.device:
        return torch.device(current_platform.device_type)

    def set_logging(self, enable: bool) -> None:
        self._enable_logging = enable

    def _run_verification(
        self,
        verification_result: VerificationResult,
        stage_name: str,
        verification_type: str,
    ) -> None:
        if verification_result.is_valid():
            return
        failed_fields = verification_result.get_failed_fields()
        if not failed_fields:
            return
        detailed_summary = verification_result.get_failure_summary()
        failed_fields_str = ", ".join(failed_fields)
        raise StageVerificationError(
            f"{verification_type.capitalize()} verification failed for {stage_name}: "
            f"Failed fields: {failed_fields_str}. Details: {detailed_summary}"
        )



def operation(function):
    """Bind a native Node.run to its shared request and error boundary."""
    @wraps(function)
    def run(self):
        slot = self.get_input(0).get(self)
        if slot.error is not None:
            return Status(StatusCode.ErrorInvalidValue)
        batch, args = slot.state, self.server_args
        context = getattr(args, 'parallel_context', None)
        error = None
        try:
            groups = context.groups if context else {}
            policy = self.execution_policy
            policy.validate_context(group_names=set(groups), world_size=context.plan.world_size if context else 1)
            if policy.should_execute(global_rank=context.global_rank if context else 0,
                                     group_names={key for key, value in groups.items() if value.is_member}):
                name = type(self).__name__
                progress = batch.extra.get('service_progress_state')
                report = getattr(progress, 'stage', None)
                if report: report(name, 'running')
                self._run_verification(self.verify_input(batch, args), name, 'input')
                self.set_logging(not batch.extra.get('runtime_progress_enabled', False))
                with StageProfiler(name, logger=logger, metrics=batch.metrics,
                                   log_stage_start_end=False,
                                   perf_dump_path_provided=batch.perf_dump_path is not None):
                    result = function(self, batch, args)
                self._run_verification(self.verify_output(result, args), name, 'output')
                slot.state = result
                if report: report(name, 'completed')
        except BaseException as exc:
            error = exc
        finally:
            self.set_logging(True)
        try:
            synchronize_stage_error(error, context)
        except BaseException as exc:
            slot.error = exc
            return Status(StatusCode.ErrorInvalidValue)
        self.get_output(0).set(slot)
        return Status.ok()
    return run
