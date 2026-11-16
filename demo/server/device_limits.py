"""Two CPU endpoints sharing model budgets and execution concurrency."""
import argparse
import os

from nndeploy.server import Server


MIB = 1024 * 1024


def uppercase(inputs):
    return {'text': inputs['text'].upper()}


def lowercase(inputs):
    return {'text': inputs['text'].lower()}


def create_server(data_dir, admin_token=None, cpu_threads=None):
    # Three slots leave room for one replacement model during Reload.
    # Memory figures are declared estimates, not limits on Python or hardware.
    server = Server(data_dir=data_dir, admin_token=admin_token,
        device_limits={'devices': {'cpu': {'max_workers': 3, 'memory_bytes': 256 * MIB,
                                         'max_concurrency': 2, 'reserved_slots': {'lowercase': 1}}}})
    for name, function in [('uppercase', uppercase), ('lowercase', lowercase)]:
        server.add(name=name, deploy=function, config={
            'inputs': {'text': {'type': 'string'}}, 'outputs': {'text': {'type': 'string'}},
            'execution': {'devices': ['cpu'], 'memory_bytes': {'cpu': 64 * MIB},
                          'temporary_memory_bytes': {'cpu': 16 * MIB},
                          **({'isolation': 'process', 'cpu_threads': cpu_threads} if cpu_threads is not None else {})},
        })
    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--data-dir', default='./resources/device-example')
    parser.add_argument('--cpu-threads', type=int, help='Limit native compute threads in isolated Workers')
    args = parser.parse_args()
    create_server(args.data_dir, os.environ.get('NNDEPLOY_ADMIN_TOKEN'), args.cpu_threads).run(host=args.host, port=args.port)
