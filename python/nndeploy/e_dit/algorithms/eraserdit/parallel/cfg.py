"""Window-scoped dual-device CFG, preserving the serial float32 merge order."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import time

import torch


def validate_cfg_parallel(server_args, batch=None):
    target = getattr(server_args.pipeline_config, "cfg_parallel_device", None)
    if target is None:
        return None
    primary, secondary = torch.device(server_args.device), torch.device(target)
    if primary.type != "cuda" or secondary.type != "cuda" or secondary.index is None:
        raise ValueError("CFG parallel requires an explicit secondary CUDA device")
    primary_index = primary.index if primary.index is not None else torch.cuda.current_device()
    if secondary.index == primary_index or secondary.index >= torch.cuda.device_count():
        raise ValueError("CFG parallel requires two distinct visible CUDA devices")
    if server_args.resolve_resource_policy().enabled or server_args.enable_torch_compile:
        raise ValueError("CFG parallel currently requires fullgpu and torch.compile disabled")
    if batch is not None and (
        getattr(batch, "transformer_cache_mode", "off") != "off"
        or getattr(batch, "cache_text_projections", False)
    ):
        raise ValueError("CFG parallel requires transformer and text projection caches disabled")
    return secondary


class EraserDiTCFGWindow:
    """Replica and worker are released on normal completion and exceptions.

    Only the negative transformer is replicated. VAE, text encoder, scheduler,
    random generator and output writer remain owned by the primary device.
    """
    def __init__(self, transformer, device):
        self.source = transformer
        self.device = device
        self.replica = None
        self.executor = None
        self.static = None
        self.setup_seconds = 0.0
        self.steps = 0

    def __enter__(self):
        if self.device is not None:
            started = time.perf_counter()
            try:
                self.replica = deepcopy(self.source).to(self.device).eval()
                torch.cuda.synchronize(self.device)
                self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eraserdit-cfg")
                self.setup_seconds = time.perf_counter() - started
            except BaseException:
                self.replica = None
                raise
        return self

    def submit(self, **kwargs):
        # The worker uses a separate thread's CUDA context. Complete the producer
        # stream before it reads latents; all model work overlaps the positive branch.
        torch.cuda.current_stream(kwargs["hidden_states"].device).synchronize()
        return self.executor.submit(self._forward, kwargs)

    def _forward(self, kwargs):
        with torch.cuda.device(self.device), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if self.static is None:
                self.static = {
                    key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                    for key, value in kwargs.items() if key not in ("hidden_states", "timestep")
                }
            result = self.replica(
                **self.static,
                hidden_states=kwargs["hidden_states"].to(self.device),
                timestep=kwargs["timestep"].to(self.device),
            )[0].float()
            torch.cuda.current_stream(self.device).synchronize()
            self.steps += 1
            return result

    def __exit__(self, *exc):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
        self.executor = self.replica = self.static = None
        return False
