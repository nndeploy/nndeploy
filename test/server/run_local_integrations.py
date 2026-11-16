"""Run Server tests with disposable local PostgreSQL, MySQL and MinIO processes.

Requires installed database executables and the matching optional Python drivers.
Uses random loopback ports, never registers system services, and deletes test data.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time


def executable(name, prefix):
    found = shutil.which(name)
    if found:
        return found
    for root in (Path('/opt/homebrew/opt'), Path('/usr/local/opt')):
        path = root / prefix / 'bin' / name
        if path.is_file():
            return str(path)
    raise RuntimeError(f'Missing {name}; install {prefix} or add its bin directory to PATH')


def port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


@contextmanager
def databases(log_dir):
    programs = {name: executable(name, package) for package, names in (
        ('postgresql@17', ('initdb', 'postgres', 'pg_isready', 'createdb')),
        ('mysql@8.4', ('mysqld', 'mysqladmin', 'mysql'))) for name in names}
    processes = []
    with tempfile.TemporaryDirectory(prefix='nndeploy-database-integration-') as temporary:
        root = Path(temporary)
        pg_port, my_port = port(), port()
        with (log_dir / 'initialize.log').open('w') as initialization, (log_dir / 'postgres.log').open('w') as pg_log, (log_dir / 'mysql.log').open('w') as my_log:
            try:
                subprocess.run([programs['initdb'], '-D', str(root/'pg'), '-U', 'nndeploy_test', '-A', 'trust', '--no-locale', '--encoding=UTF8'], check=True, stdout=initialization, stderr=initialization)
                subprocess.run([programs['mysqld'], '--no-defaults', '--initialize-insecure', '--datadir='+str(root/'mysql')], check=True, stdout=initialization, stderr=initialization)
                processes.append(subprocess.Popen([programs['postgres'], '-D', str(root/'pg'), '-h', '127.0.0.1', '-p', str(pg_port), '-k', str(root)], stdout=pg_log, stderr=pg_log))
                processes.append(subprocess.Popen([programs['mysqld'], '--no-defaults', '--datadir='+str(root/'mysql'), '--bind-address=127.0.0.1', '--port='+str(my_port), '--mysqlx=OFF', '--socket='+str(root/'mysql.sock'), '--pid-file='+str(root/'mysql.pid')], stdout=my_log, stderr=my_log))
                for _ in range(200):
                    if any(process.poll() is not None for process in processes):
                        raise RuntimeError('Test database exited; inspect database logs')
                    ready = subprocess.run([programs['pg_isready'], '-h', '127.0.0.1', '-p', str(pg_port)], capture_output=True).returncode == 0
                    ready &= subprocess.run([programs['mysqladmin'], '--no-defaults', '--socket='+str(root/'mysql.sock'), '-u', 'root', 'ping'], capture_output=True).returncode == 0
                    if ready:
                        break
                    time.sleep(.1)
                else:
                    raise RuntimeError('Test databases did not become ready')
                subprocess.run([programs['createdb'], '-h', '127.0.0.1', '-p', str(pg_port), '-U', 'nndeploy_test', 'nndeploy_test'], check=True)
                subprocess.run([programs['mysql'], '--no-defaults', '--socket='+str(root/'mysql.sock'), '-u', 'root', '-e', 'CREATE DATABASE nndeploy_test CHARACTER SET utf8mb4 COLLATE utf8mb4_bin'], check=True)
                yield [f'postgresql+psycopg://nndeploy_test@127.0.0.1:{pg_port}/nndeploy_test', f'mysql+pymysql://root@127.0.0.1:{my_port}/nndeploy_test']
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', default='logs/server-development/integration-services')
    parser.add_argument('--s3', action='store_true', help='Also start MinIO through the S3 test fixture')
    parser.add_argument('--redis', action='store_true', help='Also start Redis through the queue test fixture')
    args = parser.parse_args()
    logs = Path(args.log_dir).resolve()
    logs.mkdir(parents=True, exist_ok=True)
    with databases(logs) as urls:
        environment = dict(os.environ, NNDEPLOY_TEST_DATABASE_URLS=json.dumps(urls))
        if args.s3:
            environment['NNDEPLOY_TEST_S3'] = '1'
        if args.redis:
            environment['NNDEPLOY_TEST_REDIS'] = '1'
        return subprocess.call([sys.executable, '-m', 'pytest', 'test/server', 'test/dag/test_graph_runner_pipeline.py',
                                'test/python/test_server_regressions.py', '-q'], env=environment)


if __name__ == '__main__':
    raise SystemExit(main())
