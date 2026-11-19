"""Ulysses x Ring on orthogonal NCCL groups, with unequal-length support."""
from contextlib import nullcontext

import torch
import torch.distributed as dist

from nndeploy.e_dit.parallel.sequence import SequenceRank


class DistributedSequenceRank(SequenceRank):
    def __init__(self, groups, *, ring_mode='reference', packing='reference',
                 head_chunks=1, output_overlap=False):
        self.groups = groups
        self.degree = groups.topology.sp
        self.rank = groups.coordinates[1] + groups.topology.ulysses * groups.coordinates[2]
        self.start = self.end = 0
        self.ring_mode = ring_mode
        self.calls = dict(ulysses=0, ring=0)
        self.partial_attention_dtype = None
        self.effective_attention = None
        self.packing = packing
        chunks = head_chunks
        self.head_chunk_policy = 'auto' if chunks == 'auto' else 'fixed'
        self.head_chunks = 1 if chunks == 'auto' else int(chunks)
        self.head_overlap = True
        self.output_overlap = bool(output_overlap)
        self._communication_stream = None
        if self.head_chunks not in (1, 2, 4):
            raise ValueError('head_chunks must be auto, 1, 2 or 4')
        self.profiler = None
        if self.packing not in ('reference', 'packed', 'direct'):
            raise ValueError('packing must be reference, packed or direct')

    def _length(self, sp_rank):
        return self.length * (sp_rank + 1) // self.degree - self.length * sp_rank // self.degree

    def _region(self, name):
        return self.profiler.region(name) if self.profiler is not None else nullcontext()

    def _ulysses_input(self, value, *, qkv=None, head_offset=0, head_count=None, asynchronous=False):
        group, ranks, urank = self.groups.get('ulysses')
        size = len(ranks)
        if size == 1:
            return value
        batch, local, heads, dim = value.shape
        if qkv is not None:
            batch *= 3
        if heads % size:
            raise ValueError('attention heads must be divisible by Ulysses degree')
        h = heads // size if head_count is None else head_count
        rrank = self.groups.coordinates[2]
        lengths = [self._length(rrank * size + u) for u in range(size)]
        # Pack all destination heads in one copy instead of materializing each
        # head shard and concatenating those shards into a second allocation.
        with self._region('ulysses.input_pack'):
            if qkv is not None:
                from nndeploy.e_dit.layers.attention.qkv_packing import pack_qkv
                packed = pack_qkv(*qkv, size, head_offset=head_offset, head_count=head_count)
            elif self.packing in ('packed', 'direct'):
                packed = value.unflatten(2, (size, h)).permute(2, 0, 1, 3, 4).contiguous().view(-1)
            else:
                packed = torch.cat([x.contiguous().flatten() for x in value.split(h, dim=2)])
        recv_counts = [batch * length * h * dim for length in lengths]
        output = value.new_empty(sum(recv_counts))
        with self._region('ulysses.input_all_to_all'):
            work = dist.all_to_all_single(output, packed, recv_counts,
                                   [batch * local * h * dim] * size, group=group,
                                   **({'async_op': True} if asynchronous else {}))
            if asynchronous:
                # NCCL wait establishes a dependency on the current CUDA
                # stream; the host can enqueue the next head chunk.
                work.wait()
        self.calls['ulysses'] += 1
        with self._region('ulysses.input_unpack'):
            return torch.cat([x.view(batch, length, h, dim) for x, length in
                              zip(output.split(recv_counts), lengths)], dim=1)

    def _ulysses_output(self, value):
        group, ranks, urank = self.groups.get('ulysses')
        size = len(ranks)
        if size == 1:
            return value
        batch, _, heads, dim = value.shape
        rrank = self.groups.coordinates[2]
        lengths = [self._length(rrank * size + u) for u in range(size)]
        with self._region('ulysses.output_pack'):
            if self.packing in ('packed', 'direct') and len(set(lengths)) == 1:
                # Equal shards require only one layout copy.
                packed = value.unflatten(1, (size, lengths[0])).permute(1, 0, 2, 3, 4).contiguous().view(-1)
            else:
                packed = torch.cat([x.contiguous().flatten() for x in value.split(lengths, dim=1)])
        count = batch * lengths[urank] * heads * dim
        output = value.new_empty(count * size)
        with self._region('ulysses.output_all_to_all'):
            dist.all_to_all_single(output, packed, [count] * size,
                                   [batch * length * heads * dim for length in lengths], group=group)
        self.calls['ulysses'] += 1
        with self._region('ulysses.output_unpack'):
            return torch.cat([x.view(batch, lengths[urank], heads, dim)
                              for x in output.split(count)], dim=2)

    def _head_chunk_attention(self, query, key, value, impl, metadata):
        size = len(self.groups.get('ulysses')[1])
        if (not query.is_cuda or self.packing != 'direct' or size < 2
                or len(self.groups.get('ring')[1]) != 1 or torch.is_grad_enabled()
                or query.shape[2] % (size * self.head_chunks)):
            raise ValueError('head chunks require inference CUDA/direct Ulysses without Ring, '
                             'and heads divisible by Ulysses degree * chunks')
        heads = query.shape[2] // size // self.head_chunks
        if self.output_overlap and self.head_overlap:
            return self._head_input_output_pipeline(query, key, value, impl, metadata, size, heads)
        current = torch.cuda.current_stream(query.device)
        outputs = []
        if not self.head_overlap:
            for chunk in range(self.head_chunks):
                q, k, v = self._ulysses_input(query, qkv=(query, key, value),
                    head_offset=chunk * heads, head_count=heads).chunk(3, dim=0)
                with self._region('self_attention'):
                    outputs.append(impl.forward(q, k, v, metadata))
        else:
            if self._communication_stream is None:
                self._communication_stream = torch.cuda.Stream(device=query.device)
            communication = self._communication_stream
            communication.wait_stream(current)
            # Keep cross-stream inputs alive until both streams join. Explicit
            # retirement permits allocator reuse without record_stream's
            # deferred event polling on every QKV allocation.
            received_buffers = []
            def receive_chunk(chunk):
                with torch.cuda.stream(communication):
                    received = self._ulysses_input(query, qkv=(query, key, value),
                        head_offset=chunk * heads, head_count=heads, asynchronous=True)
                    ready = torch.cuda.Event()
                    ready.record(communication)
                received_buffers.append(received)
                return received, ready

            try:
                received, ready = receive_chunk(0)
                for chunk in range(self.head_chunks):
                    current.wait_event(ready)
                    q, k, v = received.chunk(3, dim=0)
                    with self._region('self_attention'):
                        outputs.append(impl.forward(q, k, v, metadata))
                    # Launch compute before enqueueing the next communication:
                    # preparing every chunk first delays the first SDPA launch.
                    if chunk + 1 < self.head_chunks:
                        received, ready = receive_chunk(chunk + 1)
            finally:
                # Receives belong to communication, inputs to current. Keep
                # all receives until their reads are ordered before reuse on
                # the allocating stream. Record current's event BEFORE its
                # wait on communication to avoid a cyclic dependency.
                communication.wait_stream(current)
                current.wait_stream(communication)
        self.effective_attention = type(impl).__name__ + '_head_chunks'
        # Restore the original head order before the existing output exchange.
        with self._region('ulysses.head_concat'):
            output = torch.cat(outputs, dim=2)
        return self._ulysses_output(output)

    def _head_input_output_pipeline(self, query, key, value, impl, metadata, size, heads):
        """Exchange completed heads while computing the next head chunk.

        Every rank enqueues input i+1 BEFORE output i on the same communication
        stream. Compute i+1 waits only for its input event, so output i may run
        concurrently. No collective order depends on data or completion timing.
        """
        current = torch.cuda.current_stream(query.device)
        if self._communication_stream is None:
            self._communication_stream = torch.cuda.Stream(device=query.device)
        communication = self._communication_stream
        communication.wait_stream(current)
        receives, computed, outputs = [], [], []

        def receive(index):
            with torch.cuda.stream(communication):
                received = self._ulysses_input(query, qkv=(query, key, value),
                    head_offset=index * heads, head_count=heads, asynchronous=True)
                ready = torch.cuda.Event()
                ready.record(communication)
            receives.append(received)
            return received, ready

        try:
            received, ready = receive(0)
            for chunk in range(self.head_chunks):
                current.wait_event(ready)
                q, k, v = received.chunk(3, dim=0)
                with self._region('self_attention'):
                    attended = impl.forward(q, k, v, metadata)
                computed.append(attended)
                finished = torch.cuda.Event()
                finished.record(current)
                if chunk + 1 < self.head_chunks:
                    received, ready = receive(chunk + 1)
                with torch.cuda.stream(communication):
                    communication.wait_event(finished)
                    outputs.append(self._ulysses_output(attended))
        finally:
            # Keep allocations on both streams alive until reads are ordered
            # before allocator reuse, including the exceptional path.
            communication.wait_stream(current)
            current.wait_stream(communication)
        for output in outputs:
            output.record_stream(current)
        with self._region('ulysses.head_concat'):
            # Each exchange returns [rank0 chunk_i, rank1 chunk_i, ...]. Restore
            # rank-major, then chunk-major head order before the projection.
            output = torch.stack([part.unflatten(2, (size, heads)) for part in outputs], dim=3)
            output = output.flatten(2, 4)
        self.effective_attention = type(impl).__name__ + '_head_input_output_pipeline'
        return output

    def _rotate(self, key, value, owner):
        group, ranks, rank = self.groups.get('ring')
        size = len(ranks)
        next_owner = (owner - 1) % size
        length = self.length * (next_owner + 1) // size - self.length * next_owner // size
        shape = (key.shape[0], length, key.shape[2], key.shape[3])
        recv_k, recv_v = key.new_empty(shape), value.new_empty(shape)
        key, value = key.contiguous(), value.contiguous()
        operations = [dist.P2POp(dist.isend, key, ranks[(rank + 1) % size], group),
                      dist.P2POp(dist.irecv, recv_k, ranks[(rank - 1) % size], group),
                      dist.P2POp(dist.isend, value, ranks[(rank + 1) % size], group),
                      dist.P2POp(dist.irecv, recv_v, ranks[(rank - 1) % size], group)]
        for work in dist.batch_isend_irecv(operations):
            work.wait()
        self.calls['ring'] += 1
        return recv_k, recv_v, next_owner

    def attention(self, query, key, value, impl, metadata):
        if impl.causal or getattr(impl, 'dropout', 0) or metadata.attn_mask is not None:
            raise ValueError('distributed DiT self-attention requires unmasked noncausal SDPA with dropout=0')
        if self.head_chunk_policy == 'auto':
            from nndeploy.e_dit.layers.attention.ulysses_policy import select_head_chunks
            self.head_chunks = select_head_chunks(query, self.degree, self.length,
                packing=self.packing, ring_size=len(self.groups.get('ring')[1]))
        if self.head_chunks > 1:
            return self._head_chunk_attention(query, key, value, impl, metadata)
        if len(self.groups.get('ulysses')[1]) > 1:
            # Q/K/V have identical layouts in self-attention. Batch packing
            # keeps their reduction order unchanged while using one exchange.
            if self.packing == 'direct':
                q, k, v = self._ulysses_input(query, qkv=(query, key, value)).chunk(3, dim=0)
            else:
                with self._region('ulysses.qkv_concat'):
                    packed_qkv = torch.cat((query, key, value), dim=0)
                q, k, v = self._ulysses_input(packed_qkv).chunk(3, dim=0)
                del packed_qkv
        else:
            q, k, v = query, key, value
        _, ranks, owner = self.groups.get('ring')
        if len(ranks) == 1:
            self.effective_attention = type(impl).__name__
            with self._region('self_attention'):
                output = impl.forward(q, k, v, metadata)
        elif self.ring_mode == 'reference':
            self.effective_attention = 'torch_sdpa_ring_gather'
            # Alignment path preserves full-K reduction order. Communication
            # is ring P2P, but this mode retains all KV shards; it does not claim
            # online Ring's reduced KV memory. Optimize only after SSIM gating.
            pieces = {owner: (k, v)}
            for _ in range(len(ranks) - 1):
                k, v, owner = self._rotate(k, v, owner)
                pieces[owner] = (k, v)
            full_k = torch.cat([pieces[i][0] for i in range(len(ranks))], dim=1)
            full_v = torch.cat([pieces[i][1] for i in range(len(ranks))], dim=1)
            output = impl.forward(q, full_k, full_v, metadata)
        elif self.ring_mode == 'streaming':
            self.effective_attention = 'triton_ring_fp32_fragments'
            from nndeploy.e_dit.layers.attention.ring_accumulator import piece
            if not q.is_cuda or q.shape[-1] not in (16, 32, 64):
                raise ValueError('streaming Ring requires CUDA with head dimension 16, 32 or 64')
            state = carry_k = carry_v = None
            started = finished = False
            output = None
            self.partial_attention_dtype = 'float32_fragment_accumulator'
            # Every rank sees shards in the same descending global order.
            # Different ranks start on different hops; a second circuit lets
            # all ranks finish without buffering the whole sequence.
            for hop in range(2 * len(ranks) - 1):
                if owner == len(ranks) - 1:
                    started = True
                if started and not finished:
                    chunk_k = k if carry_k is None else torch.cat((k, carry_k), dim=1)
                    chunk_v = v if carry_v is None else torch.cat((v, carry_v), dim=1)
                    begin = self.length * owner // len(ranks)
                    skip = min((-begin) % 128, chunk_k.shape[1])
                    carry_k, carry_v = chunk_k[:, :skip].clone(), chunk_v[:, :skip].clone()
                    if chunk_k.shape[1] > skip:
                        result = piece(q, chunk_k[:, skip:], chunk_v[:, skip:], state,
                                       final=owner == 0, scale=impl.softmax_scale)
                        if owner == 0:
                            output = result
                        else:
                            state = result
                    finished = owner == 0
                    del chunk_k, chunk_v
                if hop + 1 < 2 * len(ranks) - 1:
                    k, v, owner = self._rotate(k, v, owner)
            if output is None:
                raise RuntimeError('streaming Ring did not finish the global sequence')
        else:
            self.effective_attention = 'torch_flash_ring_lse_merge'
            if not q.is_cuda or metadata.attn_mask is not None:
                raise ValueError('online Ring requires unmasked CUDA attention')
            # Experimental: native partial outputs round before the FP32 LSE
            # merge. This path failed the full-video 0.985 gate; reference is
            # deliberately the default. FP16 partials also failed and were not
            # adopted (see the distributed-parallel validation report).
            output_dtype = q.dtype
            self.partial_attention_dtype = str(q.dtype)
            accumulator = total_lse = None
            for step in range(len(ranks)):
                out, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                    0., False, False, scale=impl.softmax_scale)
                if accumulator is None:
                    accumulator, total_lse = out.float(), lse
                else:
                    merged = torch.logaddexp(total_lse, lse)
                    accumulator.mul_(torch.exp(total_lse - merged).unsqueeze(-1))
                    accumulator.add_(out.float() * torch.exp(lse - merged).unsqueeze(-1))
                    total_lse = merged
                if step + 1 < len(ranks):
                    k, v, owner = self._rotate(k, v, owner)
            output = accumulator.to(output_dtype).transpose(1, 2).contiguous()
        return self._ulysses_output(output)
