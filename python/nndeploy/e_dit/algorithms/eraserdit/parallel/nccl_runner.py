"""Rank-local DiT computation; scheduler and RNG remain in the owner pipeline."""
from contextlib import nullcontext

import torch
import torch.distributed as dist

from nndeploy.e_dit.distributed.dit_groups import gather_variable
from nndeploy.e_dit.algorithms.eraserdit.parallel.nccl_sequence import DistributedSequenceRank
from nndeploy.e_dit.layers.operator_fusion.runtime import OperatorFusionRequestStats, operator_fusion_request_scope


class DiTRankRunner:
    def __init__(self, model, groups, config):
        self.model, self.groups, self.config = model, groups, config
        self.sequence = DistributedSequenceRank(groups, ring_mode=config.ring_attention_mode) if groups.topology.sp > 1 else None
        self.static = None
        self.rotary = None
        self.forwards = 0
        self.text_caches = {}
        self.cache_window = None
        self.cache_steps = 0
        self.cache_options = None
        self.text_weight_signature = None
        self.fusion_stats = OperatorFusionRequestStats(model.operator_fusion_decision)
        for block in model.transformer_blocks:
            block.attn1.processor.sequence_parallel = self.sequence

    def reset(self):
        if self.cache_window is not None:
            incomplete = self.cache_steps != self.cache_window.controller.total_steps
            self.cache_window.__exit__(RuntimeError if incomplete else None, None, None)
            self.cache_window = None
        self.cache_steps = 0
        self.cache_options = None
        self.static = self.rotary = None
        self.text_caches.clear()
        self.text_weight_signature = None
        self.fusion_stats = OperatorFusionRequestStats(self.model.operator_fusion_decision)

    def _text_weights(self):
        modules = [self.model.caption_projection]
        for block in self.model.transformer_blocks:
            modules.extend((block.attn2.to_k, block.attn2.to_v, block.attn2.norm_k))
        return tuple((id(p), p._version, p.device, p.dtype)
                     for module in modules for p in module.parameters())

    def predict(self, packet, *, owner_only=False):
        if packet['static'] is not None:
            self.reset()
            self.static = packet['static']
            options = packet.get('residual_cache') or {}
            self.cache_options = dict(options)
            if options.get('transformer_cache_mode', 'off') != 'off':
                from types import SimpleNamespace
                from nndeploy.e_dit.algorithms.eraserdit.cache.adapter import EraserDiTCacheWindow
                from nndeploy.e_dit.algorithms.eraserdit.parallel.validation import validate_nccl_text_cache
                from nndeploy.e_dit.distributed.dit_groups import DiTCacheCoordinator
                validate_nccl_text_cache(self.config, self.groups.topology)
                batch = SimpleNamespace(**dict(options, cache_text_projections=False), extra={})
                self.cache_window = EraserDiTCacheWindow(batch, total_steps=packet['total_steps'],
                    num_blocks=len(self.model.transformer_blocks), sp_degree=self.groups.topology.sp,
                    sp_rank=self.groups.get('sp')[2], cfg_degree=self.groups.topology.cfg,
                    cfg_rank=self.groups.coordinates[3],
                    coordinator=DiTCacheCoordinator(self.groups) if self.groups.topology.sp > 1 else None)
            if packet.get('cache_text_projections', False):
                from nndeploy.e_dit.algorithms.eraserdit.parallel.validation import validate_nccl_text_cache
                from nndeploy.e_dit.algorithms.eraserdit.cache.text import EraserDiTTextCache
                validate_nccl_text_cache(self.config, self.groups.topology)
                branches = ('positive', 'negative') if self.groups.topology.cfg == 1 else (
                    ('positive',) if self.groups.coordinates[3] == 0 else ('negative',))
                self.text_caches = {branch: EraserDiTTextCache() for branch in branches}
        elif bool(packet.get('cache_text_projections', False)) != bool(self.text_caches):
            raise ValueError('text cache mode cannot change within a window')
        if (packet.get('residual_cache') or {}) != self.cache_options:
            raise ValueError('residual cache options cannot change within a window')
        if self.cache_window is not None and packet.get('step') != self.cache_steps:
            raise ValueError('residual cache steps must be consecutive within a window')
        hidden, timestep = packet['hidden'], packet['timestep']
        if self.static is None:
            raise RuntimeError('first step requires window static conditions')
        if self.text_caches:
            signature = self._text_weights()
            if signature != self.text_weight_signature:
                for cache in self.text_caches.values():
                    cache.clear()
                self.text_weight_signature = signature
        topology = self.groups.topology
        cfg_rank = self.groups.coordinates[3]
        # TP and FSDP replicas must still execute every forward/collective, but
        # their identical final outputs need not be assembled and returned.
        output_rank = not owner_only or (self.groups.coordinates[0] == 0 and self.groups.coordinates[4] == 0)
        branches = ('positive', 'negative') if topology.cfg == 1 else (('positive',) if cfg_rank == 0 else ('negative',))
        outputs = {}
        fusion_scope = (operator_fusion_request_scope(self.model.operator_fusion_decision, stats=self.fusion_stats)
                        if self.model.operator_fusion_decision.effective_ops else nullcontext())
        with fusion_scope, torch.no_grad(), torch.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=hidden.is_cuda):
            for branch in branches:
                values = self.static[branch]
                if self.rotary is None:
                    self.rotary = self.model.rope(hidden, values['num_frames'], values['height'], values['width'],
                                                  values.get('rope_interpolation_scale'), values.get('video_coords'))
                scope = self.sequence.linear_scope(self.model, self.config.sp_linear_mode) if self.sequence else nullcontext()
                cache_kwargs = self.cache_window.kwargs(branch, self.cache_steps) if self.cache_window else {}
                with scope:
                    output = self.model(**values, hidden_states=hidden, timestep=timestep,
                                        image_rotary_emb=self.rotary, sequence_parallel=self.sequence,
                                        text_cache=self.text_caches.get(branch), **cache_kwargs)[0]
                self.forwards += 1
                if not output_rank:
                    continue
                if self.sequence:
                    group, ranks, _ = self.groups.get('sp')
                    lengths = tuple(self.sequence._length(i) for i in range(len(ranks)))
                    output = gather_variable(output, group, len(ranks), dim=1, lengths=lengths,
                                             dst=ranks[0] if owner_only else None)
                    if output is None:
                        continue
                    from nndeploy.e_dit.algorithms.eraserdit.models.transformer import unpack_latents
                    output = unpack_latents(output, values['num_frames'], values['height'], values['width'],
                                            self.model.config.patch_size, self.model.config.patch_size_t)
                outputs[branch] = output.float()
        self.cache_steps += 1
        if not output_rank:
            return None
        if owner_only and self.sequence and self.sequence.rank != 0:
            return None
        if topology.cfg == 2:
            group, ranks, _ = self.groups.get('cfg')
            local = outputs[branches[0]].contiguous()
            pair = [torch.empty_like(local) for _ in ranks] if not owner_only or self.groups.rank == ranks[0] else None
            if owner_only:
                dist.gather(local, gather_list=pair, dst=ranks[0], group=group)
                if pair is None:
                    return None
            else:
                dist.all_gather(pair, local, group=group)
            return pair[1], pair[0]
        return outputs['negative'], outputs['positive']

    def cache_report(self):
        if self.cache_window is None:
            return None
        window = self.cache_window
        window._record_retained_bytes()
        report = window.controller.stats()
        report.update(text_cache={branch: cache.stats() for branch, cache in self.text_caches.items()},
                      peak_retained_tensor_bytes=window._peak_retained_bytes +
                          sum(cache.stats()['peak_retained_tensor_bytes'] for cache in self.text_caches.values()),
                      cache_probe_metric=(self.cache_options or {}).get('cache_probe_metric', 'global'),
                      cache_text_projections=bool(self.text_caches), force_compute=window.force_compute,
                      experimental=True)
        return report

    def report(self):
        device = next(self.model.parameters()).device
        weight_bytes = sum((p.to_local() if hasattr(p, 'to_local') else p).numel() * p.element_size()
                           for p in self.model.parameters())
        from nndeploy.e_dit.algorithms.eraserdit.optimizations.quantization import runtime_report
        return dict(quantization=runtime_report(self.model),
                    attention_backend=self.model.transformer_blocks[0].attn1.processor.attention_backend_report(),
                    rank=self.groups.rank, coordinates=self.groups.coordinates,
                    tp_linear_mode=self.config.tp_linear_mode, sp_linear_mode=self.config.sp_linear_mode,
                    sp_linear_policy=getattr(self.sequence, 'linear_report', None),
                    additional_nccl_groups=len(self.groups.owned),
                    successful_forwards=self.forwards, local_parameter_bytes=weight_bytes,
                    text_cache={branch: cache.stats() for branch, cache in self.text_caches.items()},
                    operator_fusion=self.fusion_stats.snapshot(),
                    residual_cache=self.cache_report(),
                    cache_probe_metric=(self.cache_options or {}).get('cache_probe_metric', 'global'),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0,
                    ring_partial_dtype=self.sequence.partial_attention_dtype if self.sequence else None,
                    actual_self_attention=self.sequence.effective_attention if self.sequence else 'torch_sdpa',
                    ulysses_packing=self.sequence.packing if self.sequence else None,
                    ulysses_head_chunks=self.sequence.head_chunks if self.sequence else 1,
                    ulysses_head_chunk_policy=self.sequence.head_chunk_policy if self.sequence else 'fixed',
                    ulysses_head_overlap=bool(self.sequence and self.sequence.head_chunks > 1
                                              and self.sequence.head_overlap),
                    ulysses_output_overlap=bool(self.sequence and self.sequence.head_chunks > 1
                                                and self.sequence.head_overlap and self.sequence.output_overlap),
                    collectives=self.sequence.calls if self.sequence else {})
