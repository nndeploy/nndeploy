"""Node discovery must not terminate the editor process during graph tracing."""
import os
import subprocess
import sys
import textwrap


def test_editor_catalog_traces_registered_graphs_without_models(tmp_path):
    script = textwrap.dedent('''
        import json, queue, sys
        from pathlib import Path
        from types import SimpleNamespace
        from fastapi.testclient import TestClient
        from nndeploy.server import Server
        from nndeploy.server.template import WorkflowTemplateManager
        WorkflowTemplateManager.init_templates = staticmethod(lambda: Path(sys.argv[1]))
        server = Server(SimpleNamespace(resources=sys.argv[1], debug=True), queue.Queue(), queue.Queue(), queue.Queue(), admin_token='catalog-test')
        with TestClient(server.app) as client:
            response = client.get('/api/dag/info', headers={'Authorization': 'Bearer catalog-test'})
            assert response.status_code == 200, response.text
            nodes = response.json()['result']['nodes']
            assert any(node['id'] == 'nndeploy.basic.InputStr' for node in nodes)
            for node in nodes:
                if node['id'] == 'nndeploy::segment::SAM2Graph':
                    graph = node['nodeEntity']
                    encoder = next(child for child in graph['node_repository_'] if child['name_'] == 'encoder_infer')
                    assert [edge['name_'] for edge in encoder['outputs_']] == ['high_res_feats_0', 'high_res_feats_1', 'image_embed']
                if node['id'] == 'nndeploy::segment::SAM3Graph':
                    graph = node['nodeEntity']
                    children = {child['name_']: child for child in graph['node_repository_']}
                    assert len(children['language_encode']['outputs_']) == 1
                    assert len(children['perception_encoder']['outputs_']) == 1
                    assert len(children['detector_decoder']['outputs_']) == 4
                    assert len(children['presence_head']['outputs_']) == 4
                    assert len(children['concept_matcher']['outputs_']) == 3
            assert client.get('/api/param/types', headers={'Authorization': 'Bearer catalog-test'}).status_code == 200
            assert client.get('/health/live').status_code == 200
            print('PASS: editor catalog and enum discovery, process stays alive')
    ''')
    result = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', script, str(tmp_path)],
                            env=os.environ.copy(), capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS: editor catalog' in result.stdout
