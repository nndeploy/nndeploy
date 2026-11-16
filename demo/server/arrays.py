"""Expose a code-built Tensor Graph through the ordinary Server Deploy API."""
import argparse
from functools import partial

import numpy as np
import nndeploy.base
import nndeploy.dag
from nndeploy import _nndeploy_internal as native
from nndeploy.device import Tensor
from nndeploy.server import Server


class AddOne(nndeploy.dag.Node):
    def __init__(self, name, inputs, outputs, kind='tensor'):
        super().__init__(name, inputs, outputs)
        self.tensor = kind == 'tensor'
        self.set_input_type(native.device.Tensor if self.tensor else np.ndarray)
        self.set_output_type(native.device.Tensor if self.tensor else np.ndarray)

    def run(self):
        value = self.get_input(0).get(self)
        result = np.asarray(value) + np.float32(1)
        if self.tensor:
            result = Tensor.from_numpy(result)
            result.set_data_format(value.get_data_format())
        self.get_output(0).set(result)
        return nndeploy.base.Status.ok()


def build_graph(kind='tensor'):
    source, target = nndeploy.dag.Edge('input'), nndeploy.dag.Edge('output')
    graph = nndeploy.dag.Graph('numeric', [source], [target])
    graph.add_node(AddOne('add_one', [source], [target], kind))
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--resources', default='resources/numeric-server')
    parser.add_argument('--isolation', choices=['thread', 'process'], default='thread')
    parser.add_argument('--value-type', choices=['tensor', 'array'], default='tensor')
    parser.add_argument('--data-format', choices=['NCHW', 'NHWC'], help='Use a four-dimensional Tensor example')
    parser.add_argument('--npy', action='store_true', help='Use uploaded .npy files and return managed file IDs')
    args = parser.parse_args()
    if args.value_type == 'array' and args.data_format:
        parser.error('data-format requires value-type tensor')
    field = {'type': args.value_type, 'dtype': 'float32', 'shape': [None, 2] if args.value_type == 'tensor' else [None],
             'max_elements': 4096, 'max_bytes': 16384}
    if args.data_format:
        field.update(data_format=args.data_format,
                     shape=[None, 2, 3, 2] if args.data_format == 'NHWC' else [None, 2, 2, 3])
    sample_shape = [1 if dim is None else dim for dim in field['shape']]
    execution = {'isolation': args.isolation}
    if args.npy:
        field['media_type'] = 'application/x-npy'
    else:
        execution['warmup'] = [{'inputs': {'values': {'dtype': 'float32', 'shape': sample_shape,
                                                     'data': [0] * int(np.prod(sample_shape))}}}]
    server = Server(name='numeric', graph_factory=partial(build_graph, args.value_type), data_dir=args.resources, config={
        'inputs': {'values': dict(field, edge='input')},
        'outputs': {'values': dict(field, edge='output')},
        'execution': execution,
    })
    server.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
