"""Single-card vocab embedding helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nndeploy.e_dit.layers.utils import get_group_rank, get_group_size


DEFAULT_VOCAB_PADDING_SIZE = 64


@dataclass
class VocabParallelEmbeddingShardIndices:
    padded_org_vocab_start_index: int
    padded_org_vocab_end_index: int
    padded_added_vocab_start_index: int
    padded_added_vocab_end_index: int
    org_vocab_start_index: int
    org_vocab_end_index: int
    added_vocab_start_index: int
    added_vocab_end_index: int

    @property
    def num_org_elements(self) -> int:
        return self.org_vocab_end_index - self.org_vocab_start_index

    @property
    def num_added_elements(self) -> int:
        return self.added_vocab_end_index - self.added_vocab_start_index

    @property
    def num_org_elements_padded(self) -> int:
        return self.padded_org_vocab_end_index - self.padded_org_vocab_start_index

    @property
    def num_added_elements_padded(self) -> int:
        return self.padded_added_vocab_end_index - self.padded_added_vocab_start_index

    @property
    def num_org_vocab_padding(self) -> int:
        return self.num_org_elements_padded - self.num_org_elements

    @property
    def num_added_vocab_padding(self) -> int:
        return self.num_added_elements_padded - self.num_added_elements

    @property
    def num_elements_padded(self) -> int:
        return self.num_org_elements_padded + self.num_added_elements_padded


class UnquantizedEmbeddingMethod:
    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, **extra_weight_attrs):
        del input_size_per_partition, output_partition_sizes, input_size, output_size, extra_weight_attrs
        layer.weight = torch.nn.Parameter(torch.empty(layer.num_embeddings, layer.embedding_dim, dtype=params_dtype))
        return layer.weight

    def apply(self, layer, x, bias=None):
        del bias
        return F.linear(x, layer.weight)

    def embedding(self, layer, input_):
        return F.embedding(input_, layer.weight)


def pad_vocab_size(vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


class VocabParallelEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config=None,
        prefix: str = "",
        tp_group=None,
    ):
        super().__init__()
        del quant_config, prefix
        self.tp_group = tp_group
        self.tp_size = get_group_size(tp_group)
        self.tp_rank = get_group_rank(tp_group)
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.org_vocab_size = int(org_num_embeddings or num_embeddings)
        self.padding_size = int(padding_size)
        self.weight = torch.nn.Parameter(
            torch.empty(self.num_embeddings, self.embedding_dim, dtype=params_dtype or torch.get_default_dtype())
        )
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.weight.input_dim = 0  # type: ignore[attr-defined]
        self.weight.output_dim = 1  # type: ignore[attr-defined]
        self.weight.weight_loader = self.weight_loader  # type: ignore[attr-defined]

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if loaded_weight.shape != param.shape:
            loaded_weight = loaded_weight.reshape(param.shape)
        param.data.copy_(loaded_weight)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, self.weight)
