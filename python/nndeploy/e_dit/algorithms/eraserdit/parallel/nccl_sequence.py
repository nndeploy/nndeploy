"""EraserDiT environment policy and model projection binding for NCCL."""
import os
from nndeploy.e_dit.parallel.nccl_sequence import DistributedSequenceRank as SequenceTransport
from .mesh import SequenceRank


class DistributedSequenceRank(SequenceTransport, SequenceRank):
    def __init__(self, groups, *, ring_mode='reference'):
        overlap = os.environ.get('NNDEPLOY_DIT_ERASERDIT_ULYSSES_OUTPUT_OVERLAP', '0')
        if overlap not in ('0', '1'):
            raise ValueError('NNDEPLOY_DIT_ERASERDIT_ULYSSES_OUTPUT_OVERLAP must be 0 or 1')
        super().__init__(groups, ring_mode=ring_mode,
                         packing=os.environ.get('NNDEPLOY_DIT_ERASERDIT_NCCL_PACKING', 'reference'),
                         head_chunks=os.environ.get('NNDEPLOY_DIT_ERASERDIT_ULYSSES_HEAD_CHUNKS', '1'),
                         output_overlap=overlap == '1')
