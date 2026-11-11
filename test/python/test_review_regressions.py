"""Isolated Python regressions; no compiled extension or model download required.

Load the production function/class ASTs with dependency doubles so failures in
queue handling and metadata do not require native inference libraries.
"""
import ast
import gc
import io
import logging
import os
from pathlib import Path
import queue
import shutil
import tempfile
import threading
import traceback
from types import SimpleNamespace
import unittest
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]


def definitions(path, names, env):
    tree = ast.parse((ROOT / path).read_text())
    tree.body = [n for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(tree, path, 'exec'), env)
    return env


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code


class FileBoundaries(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'resources'
        self.root.mkdir()
        self.outside = self.root.parent / 'outside.txt'
        self.outside.write_text('untouched')
        self.env = definitions('python/nndeploy/server/files.py',
                               {'_resource_path', '_save', '_delete'}, {
            'Path': Path, 'HTTPException': HTTPException, 'UploadFile': object,
            'UploadResponse': SimpleNamespace, 'DeleteResponse': SimpleNamespace,
            'shutil': shutil, 'datetime': datetime,
        })

    def save(self, filename, directory=None):
        return self.env['_save'](SimpleNamespace(filename=filename, file=io.BytesIO(b'new')),
                                 str(directory or self.root), self.root)

    def test_normal_upload_delete(self):
        folder = self.root / 'images'
        self.assertEqual(self.save('image.png', folder).flag, 'success')
        self.assertEqual((folder / 'image.png').read_bytes(), b'new')
        self.env['_delete'](str(folder), self.root)
        self.assertFalse(folder.exists())

    def test_escape_and_root_rejected(self):
        for name in ('../outside.txt', str(self.outside), '..\\outside.txt', '', '..'):
            with self.subTest(name=name), self.assertRaises(HTTPException):
                self.save(name)
        for target in (self.outside, self.root):
            with self.assertRaises(HTTPException):
                self.env['_delete'](str(target), self.root)
        with self.assertRaises(HTTPException):
            self.save('outside.txt', self.root.parent)
        self.assertEqual(self.outside.read_text(), 'untouched')

    def test_symlink_escape_rejected(self):
        (self.root / 'link').symlink_to(self.outside)
        with self.assertRaises(HTTPException):
            self.save('link')
        with self.assertRaises(HTTPException):
            self.env['_delete'](str(self.root / 'link'), self.root)
        self.assertEqual(self.outside.read_text(), 'untouched')

    def test_delete_file_symlink_preserves_target(self):
        target = self.root / 'model.bin'
        target.write_bytes(b'model')
        link = self.root / 'shortcut.bin'
        link.symlink_to(target)
        result = self.env['_delete'](str(link), self.root)
        self.assertEqual(result.flag, 'success')
        self.assertEqual(target.read_bytes(), b'model')
        self.assertFalse(link.is_symlink())

    def test_delete_directory_symlink_preserves_contents(self):
        target = self.root / 'models'
        target.mkdir()
        (target / 'model.bin').write_bytes(b'model')
        link = self.root / 'shortcut'
        link.symlink_to(target, target_is_directory=True)
        result = self.env['_delete'](str(link), self.root)
        self.assertEqual(result.flag, 'success')
        self.assertEqual((target / 'model.bin').read_bytes(), b'model')
        self.assertFalse(link.is_symlink())

    def test_delete_dangling_symlink(self):
        link = self.root / 'shortcut'
        link.symlink_to(self.root / 'missing')
        result = self.env['_delete'](str(link), self.root)
        self.assertEqual(result.flag, 'success')
        self.assertFalse(link.is_symlink())

    def test_delete_outside_symlink_to_inside_rejected(self):
        target = self.root / 'model.bin'
        target.write_bytes(b'model')
        link = self.root.parent / 'shortcut'
        link.symlink_to(target)
        with self.assertRaises(HTTPException):
            self.env['_delete'](str(link), self.root)
        self.assertTrue(link.is_symlink())
        self.assertEqual(target.read_bytes(), b'model')


class TorchMetadata(unittest.TestCase):
    def setUp(self):
        class Base:
            def __init__(self, *args): pass
            def get_param(self): return self.param
            def set_param(self, param):
                # The native copy does not retain Python-only attributes.
                self.param = SimpleNamespace()
                return 'ok'
        env = definitions('python/nndeploy/inference/torch/torch_inference.py',
                          {'TorchInference'}, {'Inference': Base,
                          'nndeploy': SimpleNamespace(base=SimpleNamespace(Status=lambda x: x))})
        self.engine = env['TorchInference'](0)
        class Model:
            def forward(self, x): return (x, x + 1)
            __call__ = forward
        self.param = SimpleNamespace(nn_model=Model(), get_input_num=lambda: 0,
                                     get_input_name=lambda: [], get_output_num=lambda: 0,
                                     get_output_name=lambda: [])
        self.engine.set_param(self.param)

    def test_bound_forward_and_outputs(self):
        self.assertEqual(self.engine.get_num_of_input_tensor(), 1)
        self.assertEqual(self.engine.get_input_name(), 'x')
        with self.assertRaisesRegex(ValueError, 'output_num'):
            self.engine.get_num_of_output_tensor()
        self.engine.init()
        self.engine.set_input(4)
        self.engine.run()
        self.assertEqual(self.engine.get_output(), (4, 5))
        self.assertEqual(self.engine.get_num_of_output_tensor(), 2)
        self.assertEqual(self.engine.get_all_output_tensor_name(), ['0', '1'])
        self.engine.set_input(9)
        self.engine.run()
        self.assertEqual(self.engine.get_output(), (9, 10))

    def test_configured_names(self):
        self.param.get_input_name = lambda: ['image']
        self.param.get_output_name = lambda: ['boxes', 'scores']
        self.assertEqual(self.engine.get_input_name(), 'image')
        self.assertEqual(self.engine.get_output_name(1), 'scores')
        self.assertEqual(self.engine.get_num_of_output_tensor(), 2)

    def test_missing_model_and_failed_run(self):
        self.param.nn_model = None
        with self.assertRaisesRegex(ValueError, 'nn_model'):
            self.engine.init()
        self.engine.set_input(3)
        with self.assertRaises(ValueError):
            self.engine.run()
        self.assertEqual(self.engine.input_tensors_, {})
        self.assertEqual(self.engine._index, 0)
        self.assertIsNone(self.engine.get_output())


class WorkerFailures(unittest.TestCase):
    def test_exceptions_do_not_kill_worker_or_reuse_results(self):
        class StopLoop(Exception): pass
        for succeed_first in (False, True):
            with self.subTest(succeed_first=succeed_first):
                class Jobs:
                    i = 0
                    def get(self, timeout):
                        if self.i == 2: raise StopLoop()
                        self.i += 1
                        return self.i, {'id': str(self.i), 'graph_json': {}}
                class Executor:
                    def __init__(self, *args):
                        self.i = 0
                        self.runner = SimpleNamespace(get_run_status=lambda: {}, release=lambda: None)
                    def execute(self, *args):
                        self.i += 1
                        if succeed_first and self.i == 1:
                            return {}, {'previous': 'output'}, 0, 'ok'
                        raise RuntimeError('bad graph')
                noop = lambda *a, **kw: None
                env = dict(threading=threading, Empty=queue.Empty,
                           logging=SimpleNamespace(info=noop, warning=noop, error=noop),
                           os=os, traceback=traceback, gc=gc, GraphExecutor=Executor,
                           PROGRESS_INTERVAL_SEC=.01, ensure_picklable=lambda x: x,
                           nndeploy=SimpleNamespace(base=SimpleNamespace(StatusCode=SimpleNamespace(Ok=0))),
                           ExecutionStatus=lambda ok, msg: SimpleNamespace(completed=ok))
                for name in ('install_taskid_logrecord_factory', 'configure_worker_logger',
                             'redirect_fd_to_logger_once', 'poll_plugin_updates',
                             'set_task_id_fallback', 'reset_task_id', 'malloc_trim'):
                    env[name] = noop
                definitions('python/nndeploy/server/worker.py', {'run'}, env)
                results = queue.Queue()
                with self.assertRaises(StopLoop):
                    env['run'](Jobs(), results, queue.Queue(), queue.Queue(), queue.Queue(), queue.Queue(), '')
                self.assertEqual(results.qsize(), 2)
                first, second = results.get(), results.get()
                self.assertEqual(first[2], {'previous': 'output'} if succeed_first else {})
                self.assertFalse(second[1].completed)
                self.assertEqual(second[2], {})


if __name__ == '__main__':
    unittest.main()
