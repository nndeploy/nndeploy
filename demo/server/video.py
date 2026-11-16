"""Run a native video Graph as an HTTP Server, keeping its nodes resident."""
import argparse
from functools import partial

import cv2
import numpy as np
import nndeploy.base
import nndeploy.codec
import nndeploy.dag
from nndeploy.server import Server


class Resize(nndeploy.dag.Node):
    def __init__(self, name, inputs, outputs, width, height):
        super().__init__(name, inputs, outputs)
        self.set_input_type(np.ndarray)
        self.set_output_type(np.ndarray)
        self.size = (width, height)

    def run(self):
        # Replace resize with the model's per-frame processing.
        frame = self.get_input(0).get(self)
        self.get_output(0).set(cv2.resize(frame, self.size))
        return nndeploy.base.Status.ok()


def build_graph(width=320, height=240, fourcc='avc1'):
    source, target = nndeploy.dag.Edge('source'), nndeploy.dag.Edge('target')
    graph = nndeploy.dag.Graph('video')
    graph.add_node(nndeploy.codec.OpenCvVideoDecode('decode', [], [source]))
    graph.add_node(Resize('resize', [source], [target], width, height))
    encoder = nndeploy.codec.OpenCvVideoEncode('encode', [target], [])
    encoder.set_fourcc(fourcc)
    graph.add_node(encoder)
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--resources', default='resources/video-server')
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--fourcc', default='avc1', help='Encoder supported by native OpenCV (e.g. avc1 or mp4v)')
    parser.add_argument('--isolation', choices=['thread', 'process'], default='thread')
    args = parser.parse_args()
    if not 2 <= args.width <= 4096 or not 2 <= args.height <= 4096 or args.width % 2 or args.height % 2:
        parser.error('Width and height must be even numbers between 2 and 4096')
    if len(args.fourcc) != 4:
        parser.error('Fourcc must contain exactly four characters')
    field = {'type': 'file', 'media_type': 'video/mp4'}
    server = Server(name='video', data_dir=args.resources,
                    graph_factory=partial(build_graph, args.width, args.height, args.fourcc), config={
                        'inputs': {'video': dict(field, node=['decode'])},
                        'outputs': {'video': dict(field, node=['encode'])},
                        'execution': {'isolation': args.isolation},
                    })
    server.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
