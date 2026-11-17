"""Run: python demo/server/mcp_algorithms.py [--workflow workflow.json]."""
import argparse
from pydantic import BaseModel
from nndeploy.server import Server


class TextInput(BaseModel):
    text: str


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', help='Optionally export a complete Graph workflow')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    server = Server(data_dir='./resources/mcp-example', mcp_config={'enabled': True})
    server.add(name='uppercase', deploy=lambda values: {'text': values['text'].upper()},
               input_model=TextInput,
               mcp={'name': 'uppercase_text', 'description': 'Convert text to uppercase.'})
    server.add(name='count', deploy=lambda values: {'characters': len(values['text'])},
               input_model=TextInput,
               mcp={'name': 'count_characters', 'description': 'Count Unicode characters in text.'})
    if args.workflow:
        server.add(name='pipeline', workflow=args.workflow,
                   mcp={'name': 'run_pipeline', 'description': 'Execute the complete configured Graph workflow.'})
    server.run(host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
