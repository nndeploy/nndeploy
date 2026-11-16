"""Expose an ordinary audio Graph through Server's file and Deploy APIs."""
import argparse

import numpy as np
import nndeploy.base
import nndeploy.dag
from nndeploy.server import Server


class AudioGain(nndeploy.dag.Node):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, inputs, outputs)
        self.set_key('demo.server.AudioGain')
        self.set_input_type(np.ndarray)
        self.set_output_type(np.ndarray)

    def run(self):
        samples = self.get_input(0).get(self)
        self.get_output(0).set(samples * np.float32(.5))
        return nndeploy.base.Status.ok()


class AudioGainCreator(nndeploy.dag.NodeCreator):
    def create_node(self, name, inputs, outputs):
        # Retain the Python node until Graph.create_node takes ownership.
        self.node = AudioGain(name, inputs, outputs)
        return self.node


creator = AudioGainCreator()
nndeploy.dag.register_node('demo.server.AudioGain', creator)


def build_graph():
    source, target = nndeploy.dag.Edge('input'), nndeploy.dag.Edge('output')
    graph = nndeploy.dag.Graph('audio', [source], [target])
    graph.add_node(AudioGain('gain', [source], [target]))
    return graph


def config(isolation='thread'):
    field = {'type': 'array', 'dtype': 'float32', 'shape': [None],
             'media_type': 'audio/wav', 'sample_rate': 16000,
             'max_elements': 160000, 'max_bytes': 640000}
    return {'name': 'audio', 'inputs': {'audio': dict(field, edge='input')},
            'outputs': {'audio': dict(field, edge='output')}, 'execution': {'isolation': isolation}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--resources', default='resources/audio-server')
    parser.add_argument('--isolation', choices=['thread', 'process'], default='thread')
    args = parser.parse_args()
    Server(graph_factory=build_graph, config=config(args.isolation), data_dir=args.resources).run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
