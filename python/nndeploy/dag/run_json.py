"""Run a JSON workflow and report failures through the process exit status."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil

import nndeploy.base
from .node import add_global_import_lib, import_global_import_lib
from .graph_runner import GraphRunner


PARALLEL_TYPES = {
    'none': 'kParallelTypeNone', 'sequential': 'Sequential',
    'task': 'Task', 'pipeline': 'Pipeline',
}


def _parallel_type(value):
    normalized = PARALLEL_TYPES.get(value.lower(), value)
    if normalized not in PARALLEL_TYPES.values():
        raise argparse.ArgumentTypeError('Choose none, sequential, task or pipeline')
    return normalized


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json_file', '--json-file', required=True)
    parser.add_argument('--name', default='')
    parser.add_argument('--task_id', '--task-id', default='')
    parser.add_argument('--node_param', '--node-param', '-np', nargs='*', default=[],
                        help='Node parameter overrides in node:key:value format')
    parser.add_argument('--parallel_type', '--parallel-type', type=_parallel_type, default=None)
    parser.add_argument('--dump', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plugin', nargs='*', default=[])
    parser.add_argument('--resources', default='',
                        help='Copy resources into the working directory if not already present')
    args = parser.parse_args(argv)
    args.parallel_type = args.parallel_type or ''
    return args


class GraphRunnerArgs:
    def __init__(self):
        self.json_file = ''
        self.name = ''
        self.task_id = ''
        self.input_path = {}
        self.output_path = {}
        self.node_param = []
        self.parallel_type = ''
        self.dump = False
        self.debug = False
        self.plugin = []
        self.resources = ''

    def parse_args(self, argv=None):
        # Replace lists instead of accumulating values across repeated parses.
        self.__dict__.update(vars(parse_args(argv)))


def main(argv=None) -> int:
    args = GraphRunnerArgs()
    args.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format='[%(asctime)s] [%(levelname)s] %(message)s')
    logger = logging.getLogger('run_json')
    try:
        # Read before copying resources or importing plugins so missing inputs
        # fail without changing the working directory.
        graph_json = Path(args.json_file).expanduser().read_text(encoding='utf-8')
        if args.resources:
            source = Path(args.resources).expanduser().resolve()
            if not source.is_dir():
                raise ValueError(f'Resource directory not found: {source}')
            destination = Path.cwd() / source.name
            if destination.exists():
                if not destination.is_dir():
                    raise ValueError(f'Resource destination is not a directory: {destination}')
                logger.info('Using existing resources: %s', destination)
            else:
                shutil.copytree(source, destination)
        if args.plugin:
            for plugin in args.plugin:
                add_global_import_lib(str(Path(plugin).expanduser().resolve()))
            import_global_import_lib()
        runner = GraphRunner()
        _, _, status, message = runner.run(graph_json, args.name, args.task_id, args)
        if status != nndeploy.base.StatusCode.Ok:
            logger.error('Graph execution failed: %s', message)
            return 1
        logger.info('Graph execution completed successfully.')
        return 0
    except Exception:
        logger.exception('Workflow execution failed')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
