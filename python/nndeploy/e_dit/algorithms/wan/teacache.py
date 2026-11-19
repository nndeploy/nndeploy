# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from nndeploy.e_dit.algorithms.wan.teacache_params import TeaCacheParams


def _wan_1_3b_coefficients(p: TeaCacheParams) -> list[float]:
    if p.use_ret_steps:
        # from https://github.com/ali-vilab/TeaCache/blob/7c10efc4702c6b619f47805f7abe4a7a08085aa0/TeaCache4Wan2.1/teacache_generate.py#L883
        return [
            -5.21862437e04,
            9.23041404e03,
            -5.28275948e02,
            1.36987616e01,
            -4.99875664e-02,
        ]
    # from https://github.com/ali-vilab/TeaCache/blob/7c10efc4702c6b619f47805f7abe4a7a08085aa0/TeaCache4Wan2.1/teacache_generate.py#L890
    return [
        2.39676752e03,
        -1.31110545e03,
        2.01331979e02,
        -8.29855975e00,
        1.37887774e-01,
    ]


def _wan_14b_coefficients(p: TeaCacheParams) -> list[float]:
    if p.use_ret_steps:
        # from https://github.com/ali-vilab/TeaCache/blob/7c10efc4702c6b619f47805f7abe4a7a08085aa0/TeaCache4Wan2.1/teacache_generate.py#L885
        return [
            -3.03318725e05,
            4.90537029e04,
            -2.65530556e03,
            5.87365115e01,
            -3.15583525e-01,
        ]
    # from https://github.com/ali-vilab/TeaCache/blob/7c10efc4702c6b619f47805f7abe4a7a08085aa0/TeaCache4Wan2.1/teacache_generate.py#L892
    return [
        -5784.54975374,
        5449.50911966,
        -1811.16591783,
        256.27178429,
        -13.02252404,
    ]
