"""Real TCP HTTP/Worker/GPU acceptance against an existing SGLang output.

Starts an ephemeral local service, verifies byte-identical output and cancellation
recovery, then stops the service and its model processes. No deployed service is
modified. Requires an alignment run from align_sglang.py.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import httpx


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--wan-root', required=True)
    parser.add_argument('--alignment-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args=parser.parse_args()
    root=Path(args.output_dir).resolve(); root.mkdir(parents=True,exist_ok=False)
    reference=Path(args.alignment_dir).resolve()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    services = root / 'services.json'
    services.write_text(json.dumps({'services': [{
        'algorithm': 'vsr', 'name': 'vsr', 'options': {'params': {
            'model': 'wan_vsr', 'model_path': str(Path(args.checkpoint).resolve()),
            'wan_root': str(Path(args.wan_root).resolve()), 'weight_dtype': 'bf16',
            'pipeline': {key: False for key in ('cudnn_benchmark', 'channels_last_3d',
                'compile_encoder', 'compile_decoder', 'decoder_implicit_padding', 'gpu_postprocess')}
        }}
    }]}), encoding='utf-8')
    command=[sys.executable,'-u','-m','nndeploy.server','--api-only',
        '--services',str(services),'--resources',str(root),'--port',str(port)]
    results={}
    with (root/'server.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}',trust_env=False,timeout=180) as client:
                deadline=time.monotonic()+900
                while time.monotonic()<deadline:
                    if process.poll() is not None:
                        raise RuntimeError('Server startup failed; inspect server.log')
                    try:
                        if client.get('/health',timeout=1).json()=={'status':'ok'}:
                            break
                    except (httpx.HTTPError, ValueError):
                        pass
                    time.sleep(2)
                else:
                    raise TimeoutError('Server did not become ready')
                print('Server ready',flush=True)
                sampling=dict(video_input_path=str(reference/'input.mp4'),target_resolution='98x130',
                    tile_t=5,tile_h=64,tile_w=96,temporal_overlap=1,spatial_overlap=32,
                    color_ref='global',gpu_postprocess=False)
                def submit(identity,**extra):
                    response=client.post('/v1/videos/restorations',json={**sampling,'taskId':identity,**extra})
                    assert response.status_code==200 and response.json()=={'code':0,'message':'Task submitted','id':identity},response.text
                def wait(identity):
                    deadline=time.monotonic()+300
                    while time.monotonic()<deadline:
                        result=client.get(f'/v1/videos/{identity}').json()
                        if result['status'] in ('completed','failed','cancelled','expired'): return result
                        time.sleep(.2)
                    raise TimeoutError(identity)
                expected=sha(reference/'sglang/global_cpu/output.mp4')
                for identity in ('first','repeat'):
                    submit(identity)
                    if identity=='first':
                        busy=client.post('/v1/videos/restorations',json={**sampling,'taskId':'busy'}).json()
                        assert busy=={'code':2,'message':'A task is running.'},busy
                    result=wait(identity); assert result['status']=='completed',result
                    payload=client.get(f'/v1/videos/{identity}/content')
                    assert payload.status_code==200
                    digest=hashlib.sha256(payload.content).hexdigest()
                    assert digest==expected,(digest,expected)
                    results[identity]={**result,'sha256':digest}
                    print(identity,'matches SGLang',flush=True)
                submit('cancelled')
                client.delete('/v1/videos/cancelled').raise_for_status()
                result=wait('cancelled')
                assert result['status']=='cancelled' and result['error']['code']=='task_cancelled',result
                results['cancelled']=result
                submit('timeout',timeout=1)
                result=wait('timeout')
                assert result['status']=='expired' and result['error']['code']=='execution_timeout',result
                results['timeout']=result
                submit('recovered')
                result=wait('recovered'); assert result['status']=='completed',result
                digest=hashlib.sha256(client.get('/v1/videos/recovered/content').content).hexdigest()
                assert digest==expected
                results['recovered']={**result,'sha256':digest}
                loads=(root/'server.log').read_text().count(
                    'All model checkpoint weights were used when initializing WanTransformer3DModel')
                assert loads==1, f'Expected one resident model load, got {loads}'
                results['model_loads']=loads
                results['optimized']=False
                results['passed']=True
                (root/'results.json').write_text(json.dumps(results,indent=2))
                print('HTTP, cancellation, timeout and recovery passed',flush=True)
        finally:
            # Uvicorn drains requests, the common lifecycle signals cancellation,
            # and Server closes the resident Worker/process group.
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try: process.wait(timeout=200)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL); process.wait()


if __name__=='__main__':
    main()
