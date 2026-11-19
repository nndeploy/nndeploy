"""Shared runtime configuration."""

from nndeploy.e_dit.config.sampling_params import SamplingParams
from nndeploy.e_dit.config.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args,
)

__all__ = [
    "SamplingParams",
    "ServerArgs",
    "get_global_server_args",
    "set_global_server_args",
]
