"""Memory observations without resetting caller-owned CUDA peak counters."""
from pathlib import Path
import resource

import torch


def memory_observation(device):
    device = torch.device(device)
    result = dict(cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  cuda_peak_scope='since_last_external_reset')
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                result['cpu_rss_bytes'] = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    if device.type == 'cuda' and torch.cuda.is_available():
        result.update(allocated_bytes=torch.cuda.memory_allocated(device),
                      reserved_bytes=torch.cuda.memory_reserved(device),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result
