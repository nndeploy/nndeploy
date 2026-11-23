"""Request-owned scheduler preparation, with explicitly declared schedule kwargs."""
from nndeploy.e_dit.models.schedulers.request import prepare_scheduler
import torch
from .base import StateNode, operation
from .graph import register_type

@register_type
class TimestepPreparationNode(StateNode):
    def setup(self, scheduler):
        self.scheduler = scheduler

    def schedule_kwargs(self, batch, args):
        return {}

    @operation
    def run(self, batch, args):
        kwargs = self.schedule_kwargs(batch, args)
        if batch.timesteps is not None and batch.sigmas is not None:
            raise ValueError('Specify timesteps or sigmas, not both')
        if batch.timesteps is not None:
            kwargs['timesteps'] = batch.timesteps
        elif batch.sigmas is not None:
            kwargs['sigmas'] = batch.sigmas
        else:
            kwargs['num_inference_steps'] = batch.num_inference_steps
        scheduler = prepare_scheduler(self.scheduler, device=torch.device(args.device), **kwargs)
        batch.timesteps = scheduler.timesteps
        batch.extra['scheduler'] = scheduler
        return batch
