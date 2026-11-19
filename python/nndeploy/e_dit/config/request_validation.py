"""Shared sampling validation without model or tensor dependencies."""
import math


def positive_integer(value):
    return type(value) is int and value > 0


def finite_number(value, *, minimum=None, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    return minimum is None or (value > minimum if strict else value >= minimum)


def validate_generation_request(sampling, algorithm, *, integer_fps=True):
    from .capabilities import CapabilityError
    def check(key, valid, reason):
        value = getattr(sampling, key, None)
        if not valid(value):
            raise CapabilityError(algorithm, key, value, reason)
    for key in ('height', 'width', 'num_frames', 'num_inference_steps', 'num_outputs_per_prompt'):
        check(key, positive_integer, 'must be a positive integer')
    check('fps', positive_integer if integer_fps else
          lambda v: finite_number(v, minimum=0, strict=True), 'must be positive and finite')
    check('seed', lambda v: type(v) is int and 0 <= v < 2**63, 'must be an integer in [0, 2**63)')
    check('guidance_scale', lambda v: finite_number(v, minimum=0), 'must be finite and nonnegative')
    prompts = sampling.prompt if isinstance(sampling.prompt, list) else [sampling.prompt]
    if not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
        raise CapabilityError(algorithm, 'prompt', sampling.prompt, 'must contain nonempty text')
    negative = sampling.negative_prompt
    if not isinstance(negative, str) and not (
        isinstance(negative, list) and len(negative) == len(prompts)
        and all(isinstance(p, str) for p in negative)
    ):
        raise CapabilityError(algorithm, 'negative_prompt', negative, 'must be text or a matching text batch')
