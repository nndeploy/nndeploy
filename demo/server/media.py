"""Expose an ordinary byte-port Graph as a file/audio/video HTTP Server."""
import argparse

import nndeploy.base
import nndeploy.dag
from nndeploy.server import Server


class CopyFile(nndeploy.dag.Node):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, inputs, outputs)
        self.set_input_type(bytes)
        self.set_output_type(bytes)

    def run(self):
        # Replace this operation with the algorithm; Graph receives file bytes,
        # while Server owns upload validation, task records and output files.
        self.get_output(0).set(self.get_input(0).get(self))
        return nndeploy.base.Status.ok()


def build_graph():
    source, target = nndeploy.dag.Edge('input'), nndeploy.dag.Edge('output')
    graph = nndeploy.dag.Graph('media', [source], [target])
    graph.add_node(CopyFile('copy', [source], [target]))
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--resources', default='resources/media-server')
    parser.add_argument('--media-type', choices=['audio/wav', 'video/mp4', 'application/octet-stream'], default='audio/wav')
    parser.add_argument('--isolation', choices=['thread', 'process'], default='thread')
    args = parser.parse_args()
    field = {'type': 'file', 'media_type': args.media_type}
    server = Server(name='media', graph_factory=build_graph, data_dir=args.resources, config={
        'inputs': {'file': dict(field, edge='input')},
        'outputs': {'file': dict(field, edge='output')},
        'execution': {'isolation': args.isolation},
    })
    server.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
