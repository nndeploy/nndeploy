"""Server regressions that run without the native inference extension."""
import importlib.util
import logging
import os
from pathlib import Path
import queue
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_review_regressions import ROOT, definitions

spec = importlib.util.spec_from_file_location('review_task_queue', ROOT / 'python/nndeploy/server/task_queue.py')
task_queue = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = task_queue
spec.loader.exec_module(task_queue)


class QueueRegressions(unittest.TestCase):
    def setUp(self):
        self.queue = task_queue.TaskQueue(SimpleNamespace(notify_task_done=lambda *args: None), queue.Queue())

    def test_identical_timestamps_are_fifo(self):
        with patch.object(task_queue.time, 'time', return_value=100):
            self.queue.put({'id': 'first'}, 1)
            self.queue.put({'id': 'second'}, 1)
            self.queue.put({'id': 'urgent'}, 0)
        self.assertEqual([self.queue.get()[1]['id'] for _ in range(3)], ['urgent', 'first', 'second'])

    def test_pending_snapshot_and_lookup(self):
        payload = {'id': 'first', 'graph_json': {'name': 'original'}}
        self.queue.put(payload)
        payload['graph_json']['name'] = 'mutated'
        pending = self.queue.get_current_queue()['PENDING']
        self.assertEqual(pending[0]['task']['id'], 'first')
        self.assertEqual(pending[0]['state'], 'PENDING')
        self.assertEqual(self.queue.get_task_by_id('first')['task']['graph_json']['name'], 'original')
        idx, _ = self.queue.get()
        # Dequeued but blocked on dispatch must still be visible.
        self.assertEqual(self.queue.get_current_queue()['PENDING'][0]['idx'], idx)

    def test_late_dispatch_does_not_regress_running(self):
        self.queue.put({'id': 'first'})
        idx, _ = self.queue.get()
        self.queue.mark_started('first', 123)
        self.queue.mark_dispatched(idx)
        state = self.queue.get_current_queue()
        self.assertEqual(state['DISPATCHED'], [])
        self.assertEqual(state['RUNNING'][0]['worker_pid'], 123)

    def test_history_snapshot_does_not_mutate_queue(self):
        self.queue.put({'id': 'first'})
        idx, _ = self.queue.get()
        self.queue.task_done(idx, task_queue.ExecutionStatus(True), {}, {})
        history = self.queue.get_history()
        history['first']['task']['id'] = 'changed'
        self.assertEqual(self.queue.get_task_by_id('first')['task']['id'], 'first')


@unittest.skipUnless(importlib.util.find_spec('pydantic'), 'pydantic is not installed')
class SchemaRegressions(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            'review_schemas', ROOT / 'python/nndeploy/server/schemas.py')
        self.schemas = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.schemas
        spec.loader.exec_module(self.schemas)

    def test_pending_queue_matches_response_schema(self):
        q = task_queue.TaskQueue(SimpleNamespace(), queue.Queue())
        q.put({'id': 'queued'})
        state = q.get_current_queue()
        response = self.schemas.QueueStateResult(
            running=state['RUNNING'], pending=state['PENDING'], dispatched=state['DISPATCHED'])
        self.assertEqual(response.model_dump()['pending'][0]['task']['id'], 'queued')

    def test_preview_literal_validates(self):
        from pydantic import ValidationError
        self.assertEqual(self.schemas.PreviewPayload(type='preview', data={}).type, 'preview')
        with self.assertRaises(ValidationError):
            self.schemas.PreviewPayload(type='wrong', data={})


class DownloadRegressions(unittest.TestCase):
    def setUp(self):
        self.env = definitions('python/nndeploy/server/utils.py',
                               {'_atomic_write_from_stream', '_download_with_retries'}, {
            'requests': SimpleNamespace(Response=object), 'Path': Path,
            'tempfile': tempfile, 'os': os, 'logging': logging,
            'sleep': lambda *args: None,
        })
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'model.bin'
        self.path.write_bytes(b'previous complete model')

    def test_stream_failure_cleans_temp_and_preserves_destination(self):
        def chunks(**kwargs):
            yield b'partial'
            raise OSError('disconnected')
        with self.assertRaises(OSError):
            self.env['_atomic_write_from_stream'](SimpleNamespace(iter_content=chunks), self.path)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        self.assertEqual(self.path.read_bytes(), b'previous complete model')

    def test_replace_failure_cleans_temp(self):
        response = SimpleNamespace(iter_content=lambda **kwargs: iter([b'new']))
        with patch.object(os, 'replace', side_effect=OSError('rename failed')):
            with self.assertRaises(OSError):
                self.env['_atomic_write_from_stream'](response, self.path)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        self.assertEqual(self.path.read_bytes(), b'previous complete model')

    def test_retry_closes_responses_and_preserves_valid_file(self):
        responses = []
        class Response:
            def __init__(self): self.closed = False
            def __enter__(self): return self
            def __exit__(self, *args): self.closed = True
            def raise_for_status(self): raise OSError('HTTP error')
        def get(*args, **kwargs):
            response = Response()
            responses.append(response)
            return response
        self.env['requests'].get = get
        with self.assertRaises(OSError):
            self.env['_download_with_retries']('https://example.invalid/model', self.path, retries=2)
        self.assertEqual(len(responses), 2)
        self.assertTrue(all(response.closed for response in responses))
        self.assertEqual(self.path.read_bytes(), b'previous complete model')

    def test_success_replaces_complete_file(self):
        response = SimpleNamespace(iter_content=lambda **kwargs: iter([b'new', b'', b' model']))
        self.env['_atomic_write_from_stream'](response, self.path)
        self.assertEqual(self.path.read_bytes(), b'new model')
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])


if __name__ == '__main__':
    unittest.main()
