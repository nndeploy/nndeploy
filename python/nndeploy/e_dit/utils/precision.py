"""Shared dtype names for component configuration."""
import torch
PRECISION_TO_TYPE = {'fp32': torch.float32, 'float32': torch.float32,
                     'fp16': torch.float16, 'float16': torch.float16,
                     'bf16': torch.bfloat16, 'bfloat16': torch.bfloat16}
