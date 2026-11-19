"""Install nndeploy with one algorithm's dependency versions and pip indexes.

Example from the repository root:
    python python/install.py nndeploy/e_dit/algorithms/eraserdit/requirements.txt --editable
Relative requirements paths are resolved against this script's directory.
A directory path selects the requirements.txt file inside that directory.
Additional arguments, such as --dry-run, are forwarded to pip install.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('requirements', nargs='?', help='Requirements file or directory, relative to install.py or absolute')
    selection.add_argument('--algorithm', help='Directory name under e_dit/algorithms')
    parser.add_argument('-e', '--editable', action='store_true', help='Install nndeploy in editable mode')
    args, pip_args = parser.parse_known_args(argv)
    root = Path(__file__).resolve().parent
    environment = os.environ.copy()
    if args.requirements:
        requirements = (root / Path(args.requirements).expanduser()).resolve()
        if requirements.is_dir():
            requirements = requirements / 'requirements.txt'
        environment.pop('NNDEPLOY_DIT_ALGORITHM', None)
        environment['NNDEPLOY_REQUIREMENTS_OVERRIDE'] = str(requirements)
    else:
        if not re.fullmatch(r'[a-z][a-z0-9_]*', args.algorithm):
            parser.error('algorithm must be a directory name, such as eraserdit')
        requirements = root / 'nndeploy' / 'e_dit' / 'algorithms' / args.algorithm / 'requirements.txt'
        if environment.get('NNDEPLOY_REQUIREMENTS_OVERRIDE'):
            parser.error('unset NNDEPLOY_REQUIREMENTS_OVERRIDE when selecting --algorithm')
        environment['NNDEPLOY_DIT_ALGORITHM'] = args.algorithm
    if not requirements.is_file():
        parser.error(f'requirements file not found: {requirements}')
    # setup.py replaces the default pins before pip resolves this transaction.
    # Reading the same file via -r also supplies its CUDA/custom wheel indexes.
    command = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements)]
    if args.editable:
        command.append('--editable')
    command.append(str(root))
    command.extend(pip_args)
    print(f'Installing nndeploy with requirements: {requirements}', flush=True)
    return subprocess.run(command, cwd=root, env=environment).returncode


if __name__ == '__main__':
    raise SystemExit(main())
