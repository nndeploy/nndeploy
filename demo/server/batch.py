"""Combine independent HTTP tasks in one NumPy computation; no model download."""
import argparse

import numpy as np
from pydantic import BaseModel, Field

from nndeploy.server import Server


class Inputs(BaseModel):
    values: list[float] = Field(min_length=1, max_length=4096)


class Outputs(BaseModel):
    total: float
    batch_size: int


class VectorSum:
    def batch_key(self, inputs):
        # This algorithm stacks equally shaped vectors. Padding-capable models
        # can declare a different compatible key. Identity/parameters are always
        # included by Server, even if the algorithm does not include them here.
        return len(inputs['values'])

    def deploy_batch(self, items):
        matrix = np.asarray([item['inputs']['values'] for item in items], dtype=np.float64)
        totals = matrix.sum(axis=1)
        return [{'total': float(total), 'batch_size': len(items)} for total in totals]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--data-dir', default='./resources/batch-example')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--batch-wait-ms', type=float, default=20)
    args = parser.parse_args()
    server = Server(name='vectors', handler_factory=VectorSum, input_model=Inputs, output_model=Outputs,
        config={'execution': {'batching': {'enabled': True, 'max_size': args.batch_size, 'max_wait_ms': args.batch_wait_ms}}},
        data_dir=args.data_dir)
    server.run(host=args.host, port=args.port)
