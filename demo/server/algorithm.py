"""Algorithm-owned HTTP protocol: python demo/server/algorithm.py."""
import argparse
import os

from nndeploy.server import Server


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--data-dir', default='./resources/algorithm-example')
    args = parser.parse_args()
    server = Server(data_dir=args.data_dir, admin_token=os.environ.get('NNDEPLOY_ADMIN_TOKEN'))
    server.register('text', 'text.server:create_entry')
    server.add(name='text', algorithm='text', options={'uppercase': True})
    server.run(host=args.host, port=args.port)
