"""Receive incremental input and yield output within one existing Server Task."""
import argparse

from nndeploy.server import Server


async def text(inputs, *, input_stream):
    result = inputs['prefix']
    async for chunk in input_stream:
        if not isinstance(chunk, dict) or not isinstance(chunk.get('text'), str):
            raise ValueError('Each input chunk must contain a text string')
        result += chunk['text']
        yield {'text': result}
    yield {'text': result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--data-dir', default='./resources/duplex-example')
    args = parser.parse_args()
    server = Server(name='text', deploy=text, data_dir=args.data_dir, config={
        'inputs': {'prefix': {'type': 'string', 'required': False, 'default': ''}},
        'outputs': {'text': {'type': 'string'}},
        'streaming': {'enabled': True, 'duplex': True, 'max_pending_inputs': 64, 'max_duration_seconds': 300},
        'execution': {'scheduling': 'engine', 'cooperative_cancel': True, 'max_concurrency_per_worker': 4},
    })
    server.run(host=args.host, port=args.port)
