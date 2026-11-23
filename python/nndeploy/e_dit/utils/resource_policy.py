"""Compatibility exports for resource configuration and tensor operations."""

from nndeploy.e_dit.config.resource_policy import (
    RuntimeResourcePolicy,
    resolve_runtime_resource_policy,
)
from nndeploy.e_dit.memory.tensor_ops import (
    maybe_pin_tensor,
    module_device,
    module_dtype,
    move_module_to_device,
    pin_module_cpu_memory,
)
