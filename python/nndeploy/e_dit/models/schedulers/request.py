"""Construct isolated scheduler state, preserving algorithm-specific schedules."""
from copy import deepcopy
import inspect


def prepare_scheduler(template, *, device, num_inference_steps=None,
                      timesteps=None, sigmas=None, **schedule_kwargs):
    supplied = sum(value is not None for value in (num_inference_steps, timesteps, sigmas))
    if supplied != 1:
        raise ValueError('Specify exactly one of num_inference_steps, timesteps or sigmas')
    if num_inference_steps is not None and (type(num_inference_steps) is not int or num_inference_steps <= 0):
        raise ValueError('num_inference_steps must be a positive integer')
    scheduler = deepcopy(template)
    if scheduler is template:
        raise ValueError('Scheduler must support independent request copies')
    kwargs = dict(schedule_kwargs)
    for key, value in (('num_inference_steps', num_inference_steps), ('timesteps', timesteps), ('sigmas', sigmas)):
        if value is not None:
            kwargs[key] = value
    signature = inspect.signature(scheduler.set_timesteps)
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())
    if 'device' in signature.parameters or accepts_kwargs:
        kwargs['device'] = device
    unsupported = set(kwargs) - set(signature.parameters)
    if unsupported and not accepts_kwargs:
        raise ValueError(f'Scheduler does not support {sorted(unsupported)}')
    scheduler.set_timesteps(**kwargs)
    return scheduler
