"""Run a code entry with incremental outputs: python demo/server/stream.py."""
import argparse
import asyncio
import time

from pydantic import BaseModel, Field

from nndeploy.server import Server


class Inputs(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class Outputs(BaseModel):
    text: str


def words(inputs):
    parts = []
    for word in inputs['text'].split():
        parts.append(word)
        yield {'text': ' '.join(parts)}
        time.sleep(.05)


async def async_words(inputs):
    parts = []
    for word in inputs['text'].split():
        parts.append(word)
        yield {'text': ' '.join(parts)}
        await asyncio.sleep(.05)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--data-dir', default='./resources/stream-example')
    parser.add_argument('--engine-concurrency', type=int, help='Run an async engine with this many concurrent requests')
    args = parser.parse_args()
    config = {'streaming': {'enabled': True}}
    if args.engine_concurrency is not None:
        config['execution'] = {'scheduling': 'engine', 'max_concurrency_per_worker': args.engine_concurrency}
    server = Server(name='words', deploy=async_words if args.engine_concurrency is not None else words,
                    input_model=Inputs, output_model=Outputs, config=config, data_dir=args.data_dir)
    server.run(host=args.host, port=args.port)
