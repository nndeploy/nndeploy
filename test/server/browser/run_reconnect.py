"""Run a real Chrome recovery check against a disposable local editor."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='logs/server-development')
    parser.add_argument('--playwright', help='Installed Playwright module path')
    parser.add_argument('--chrome', help='Chrome executable path')
    parser.add_argument('--scenario', choices=['reconnect', 'nested', 'resources'], default='reconnect')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='editor-reconnect-', dir=output))
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    state = root / 'state.json'
    state.write_text(json.dumps({'root': str(root), 'port': port}))
    environment = dict(os.environ, NNDEPLOY_BROWSER_ROOT=str(root))
    if args.scenario == 'nested':
        environment['NNDEPLOY_BROWSER_NESTED'] = '1'
    if args.scenario == 'resources':
        environment.update(NNDEPLOY_BROWSER_RESOURCES='1', NNDEPLOY_BROWSER_ADMIN='browser-resource-test')
    if args.playwright:
        environment['NNDEPLOY_PLAYWRIGHT'] = str(Path(args.playwright).resolve())
    if args.chrome:
        environment['NNDEPLOY_CHROME'] = str(Path(args.chrome).resolve())
    with (root / 'server.log').open('w') as log:
        command = [sys.executable, str(Path(__file__).with_name('launch_reconnect.py')),
            '--host', '127.0.0.1', '--port', str(port), '--resources', str(root / 'resources'),
            '--log', str(root / 'editor.log'), '--front-end-dir', str(repo / 'app/workflow/dist')]
        if args.scenario == 'resources':
            command += ['--admin-token-env', 'NNDEPLOY_BROWSER_ADMIN']
        server = subprocess.Popen(command,
            env=environment, stdout=log, stderr=log, cwd=repo)
        try:
            deadline = time.monotonic() + 30
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Editor exited; inspect {root / "server.log"}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=1) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError('Editor did not become ready')
                time.sleep(.1)
            with (root / 'browser.log').open('w') as browser_log:
                script = {'nested': 'editor_nested.cjs', 'resources': 'editor_resources.cjs'}.get(args.scenario, 'editor_reconnect.cjs')
                result = subprocess.run(['node', str(Path(__file__).with_name(script)), str(state)],
                    cwd=repo, env=environment, stdout=browser_log, stderr=browser_log, timeout=120)
            print((root / 'browser.log').read_text(), end='')
            return result.returncode
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            print(f'Browser evidence: {root}', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
