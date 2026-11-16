"""Disposable loopback editor for browser recovery checks; never a production entrypoint."""
import json
import copy
import os
from pathlib import Path

from PIL import Image
from starlette.responses import JSONResponse

from nndeploy.server import app
from nndeploy.server.server import Server
from nndeploy.server.template import WorkflowTemplateManager


if __name__ == '__main__':
    root = Path(os.environ['NNDEPLOY_BROWSER_ROOT']).resolve()
    resources = root / 'resources'
    plugins = resources / 'plugin'
    plugins.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (20, 24), (20, 140, 220)).save(resources / 'input.png')
    (plugins / 'browser_output.py').write_text('''
import time
from pathlib import Path
import nndeploy.dag
import nndeploy.base
from nndeploy.codec.pil import PILImageEncodec
root = Path(__file__).resolve().parents[2]
class BrowserOutput(PILImageEncodec):
    def run(self):
        with (root / 'executions.txt').open('a') as stream:
            stream.write('run\\n')
        deadline = time.monotonic() + 30
        while (root / 'hold').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('Browser test did not release the Graph')
            time.sleep(.02)
        if self.path_.endswith('right.png'):
            image = self.get_input(0).get(self).copy()
            image.paste((220, 80, 20), (0, 0, image.width, image.height))
            image.save(self.path_)
            return nndeploy.base.Status.ok()
        return super().run()
class Creator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        self.nodes = []
    def create_node(self, name, inputs, outputs):
        node = BrowserOutput(name, inputs, outputs)
        node.set_key('test.browser.Output')
        self.nodes.append(node)
        return node
creator = Creator()
nndeploy.dag.register_node('test.browser.Output', creator)
''')
    graph = dict(key_='nndeploy.dag.Graph', name_='browser-reconnect',
        parallel_type_='kParallelTypeSequential', node_repository_=[
            dict(key_='nndeploy.codec.PILImageDecodec', name_='decode',
                 path_=str(resources / 'input.png'), color_mode_='RGB', inputs_=[],
                 outputs_=[dict(name_='image', type_='PIL.Image.Image')]),
            dict(key_='test.browser.Output', name_='encode',
                 path_=str(resources / 'output.png'), node_type_='Output',
                 io_type_='Image', io_params_=['path_'],
                 inputs_=[dict(name_='image', type_='PIL.Image.Image')], outputs_=[]),
        ])
    if os.environ.get('NNDEPLOY_BROWSER_NESTED') == '1':
        groups = []
        for name in ('left', 'right'):
            leaf = copy.deepcopy(graph['node_repository_'][1])
            leaf['path_'] = str(resources / (name + '.png'))
            groups.append(dict(key_='nndeploy.dag.Graph', name_=name, is_graph_=True,
                parallel_type_='kParallelTypeSequential',
                inputs_=[dict(name_='image', type_='PIL.Image.Image')], outputs_=[],
                node_repository_=[leaf]))
        graph['name_'] = 'nested-previews'
        graph['node_repository_'] = [graph['node_repository_'][0], *groups]
    if os.environ.get('NNDEPLOY_BROWSER_RESOURCES') == '1':
        import zipfile
        (root / 'weights.bin').write_bytes(b'weights')
        (root / 'replacement.bin').write_bytes(b'updated')
        (root / 'bad.zip').write_bytes(b'not a ZIP archive')
        with zipfile.ZipFile(root / 'vocab.zip', 'w') as archive:
            archive.writestr('vocab.txt', 'tokens')
        (plugins / 'model_resource.py').write_text('''
import json
from pathlib import Path
import nndeploy.base
import nndeploy.dag
class Resource(nndeploy.dag.Node):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, inputs, outputs)
        self.set_key('test.browser.Resource')
        self.set_io_type(nndeploy.dag.IOType.Model)
        self.add_io_param('artifacts')
        self.set_output_type(str)
        self.artifacts = ['']
    def run(self):
        paths = self.artifacts if isinstance(self.artifacts, list) else [self.artifacts]
        self.get_output(0).set('|'.join((Path(value) / 'vocab.txt' if Path(value).is_dir() else Path(value)).read_text() for value in paths))
        return nndeploy.base.Status.ok()
    def serialize(self):
        value = json.loads(super().serialize())
        value['artifacts.v1' if self.get_name() == 'vocab' else 'artifacts'] = self.artifacts
        return json.dumps(value)
    def deserialize(self, value):
        data = json.loads(value)
        self.artifacts = data.get('artifacts.v1', data.get('artifacts'))
        return super().deserialize(value)
class Creator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        self.nodes = []
    def create_node(self, name, inputs, outputs):
        node = Resource(name, inputs, outputs)
        self.nodes.append(node)
        return node
creator = Creator()
nndeploy.dag.register_node('test.browser.Resource', creator)
''')
        nodes = []
        for name, kind, value in [('model', 'Model', ['']), ('vocab', 'Dir', '')]:
            edge = dict(name_=name + '_text', type_='str')
            field = 'artifacts.v1' if name == 'vocab' else 'artifacts'
            nodes.append(dict(key_='test.browser.Resource', name_=name, node_type_='Intermediate',
                io_type_=kind, io_params_=[field], inputs_=[], outputs_=[edge], **{field: value}))
            nodes.append(dict(key_='nndeploy.basic.ConsolePrintNode', name_='sink_' + name,
                node_type_='Output', io_type_='String', inputs_=[edge], outputs_=[],
                frontend_params=dict(echo_to_stdout=False, print_to_logger=False, save_to_file=False)))
        graph = dict(key_='nndeploy.dag.Graph', name_='model-resources',
                     parallel_type_='kParallelTypeSequential', node_repository_=nodes)
    (root / 'graph.json').write_text(json.dumps(graph))
    WorkflowTemplateManager.init_templates = staticmethod(lambda: root)
    original = Server.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)

        @self.app.middleware('http')
        async def fault(request, call_next):
            if request.url.path == '/__test__/disconnect' and request.method == 'POST':
                sockets = list(self.sockets)
                for socket in sockets:
                    await socket.close(code=1012)
                return JSONResponse({'closed': len(sockets)})
            if request.url.path == '/api/queue' and request.method == 'POST':
                with (root / 'posts.jsonl').open('a') as stream:
                    stream.write(json.dumps({'key': request.headers.get('Idempotency-Key'),
                                             'query': request.url.query}) + '\n')
                response = await call_next(request)
                if (root / 'drop-after').exists():
                    (root / 'drop-after').unlink()
                    return JSONResponse({}, status_code=503)
                return response
            if request.url.path.startswith('/api/queue/task/') and (root / 'hide-status').exists():
                return JSONResponse({}, status_code=503)
            return await call_next(request)

    Server.__init__ = initialize
    app.main()
