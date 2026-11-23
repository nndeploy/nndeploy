"""Opt-in, one-step-per-window CPU/CUDA traces, never used for speed claims."""

from nndeploy.e_dit.utils.env import get_env
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path

import torch


class DiTStepProfiler:
    def __init__(self, rank, device):
        directory = get_env('NNDEPLOY_DIT_PROFILE_DIR')
        self.directory = Path(directory).expanduser().resolve() if directory else None
        self.target_step = int(get_env('NNDEPLOY_DIT_PROFILE_STEP', '2'))
        if self.target_step < 1:
            raise ValueError('NNDEPLOY_DIT_PROFILE_STEP must be positive (one-based)')
        self.rank, self.device = rank, torch.device(device)
        self.step = self.sample = 0
        self.active = False
        self.artifacts = []

    def reset(self):
        self.step = 0
        self.artifacts.clear()

    def region(self, name):
        return torch.profiler.record_function('dit.' + name) if self.active else nullcontext()

    @contextmanager
    def capture(self, model):
        self.step += 1
        if self.directory is None or self.step != self.target_step:
            yield
            return
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        handles, scopes = [], []

        def before(name):
            def hook(module, args):
                scope = torch.profiler.record_function('dit.module.' + name)
                scope.__enter__()
                scopes.append(scope)
            return hook

        def after(module, args, output):
            scopes.pop().__exit__(None, None, None)

        # Leaves identify projections/norms; block/attention/FFN parents provide
        # inclusive intervals. These intervals overlap and must not be summed.
        for name, module in model.named_modules():
            if name:
                handles.append(module.register_forward_pre_hook(before(name)))
                handles.append(module.register_forward_hook(after, always_call=True))
        try:
            with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
                self.active = True
                try:
                    with self.region('forward_and_output_gather'):
                        yield
                finally:
                    self.active = False
            self.directory.mkdir(parents=True, exist_ok=True)
            self.sample += 1
            stem = self.directory / f'rank{self.rank}_pid{os.getpid()}_sample{self.sample}'
            trace = str(stem) + '.trace.json'
            profile.export_chrome_trace(trace)
            rows = [dict(name=e.key, device_type=str(e.device_type), calls=e.count, cpu_total_us=e.cpu_time_total,
                         cpu_self_us=e.self_cpu_time_total,
                         device_total_us=e.device_time_total,
                         device_self_us=e.self_device_time_total)
                    for e in profile.key_averages()]
            summary = str(stem) + '.summary.json'
            Path(summary).write_text(json.dumps(dict(rank=self.rank, step=self.step,
                diagnostic_only=True, note='Inclusive intervals overlap; ranks run concurrently. '
                'Profiler overhead is included. CPU and CUDA annotations can share a name; '
                'filter device_type before aggregating. Device time is not end-to-end latency.',
                events=rows), indent=2) + '\n')
            self.artifacts.append(dict(trace=trace, summary=summary, step=self.step))
        finally:
            self.active = False
            for handle in handles:
                handle.remove()
