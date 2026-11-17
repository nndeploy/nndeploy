# executor.py

import requests
import os
import logging
import json
import threading
from typing import Dict, Tuple, Any, List
from pathlib import Path
from urllib.parse import urlparse

from nndeploy.dag import GraphRunner

class GraphExecutor:
    """Encapsulate nndeploy load and run logic"""
    def __init__(self, resources, cache_type=False, cache_size=None):
        self.runner = GraphRunner()
        self.resources = resources

    def interrupt_running(self):
        self.runner.cancel_running()

    def execute(self, graph_json: Dict, task_id: str, *, cpu=None) -> Tuple[Dict, float]:
        self._validate_codec_inputs(graph_json)
        name = graph_json.get("name_")
        if isinstance(graph_json, (dict, list)):
            graph_json = json.dumps(graph_json, ensure_ascii=False)

        callbacks = dict(before_init=cpu.configure_graph, before_run=cpu.apply) if cpu is not None else {}
        return self.runner.run(graph_json, name, task_id, **callbacks)

    @staticmethod
    def _validate_codec_inputs(graph_json, parents=()):
        """Report local codec input mistakes before native loading loses context."""
        codecs = {
            'nndeploy::codec::OpenCvImageDecode': 'file',
            'nndeploy::codec::OpenCvImagesDecode': 'directory',
            'nndeploy::codec::OpenCvVideoDecode': 'file',
        }
        for node in graph_json.get('node_repository_', []):
            names = (*parents, node.get('name_', '<unnamed>'))
            expected = codecs.get(node.get('key_'))
            path = node.get('path_')
            # Video streams are opened and validated by the native codec.
            stream = node.get('key_') == 'nndeploy::codec::OpenCvVideoDecode' and isinstance(path, str) and '://' in path
            if expected and isinstance(path, str) and not stream:
                valid = bool(path) and (Path(path).is_dir() if expected == 'directory' else Path(path).is_file())
                if not valid:
                    hint = ' For a single image, use OpenCvImageDecode.' if expected == 'directory' else ''
                    if path != path.strip():
                        hint += ' Check leading or trailing whitespace in the path.'
                    raise ValueError(f"Node {'/'.join(names)!r}: path_ must be an existing {expected}; got {path!r}.{hint}")
            GraphExecutor._validate_codec_inputs(node, names)

    def handle_urls(self, graph_json: Dict) -> Dict[str, str]:

        url_fields = {
            "image_url_": "images",
            "video_url_": "videos",
            "audio_url_": "audios",
            "model_url_": "models",
            "other_url_": "others"
        }

        result_paths = {}

        for field, subdir in url_fields.items():
            if field not in graph_json:
                continue

            url_list = graph_json[field]
            if not isinstance(url_list, list):
                logging.warning(f"{field} is not a list")
                continue
            if not url_list:
                logging.info(f"{field} is an empty list, skipping.")
                continue

            for idx, url in enumerate(url_list):
                if not isinstance(url, str):
                    logging.warning(f"{field}.{idx} is not a string.")
                    continue

                try:
                    type_, uri = url.split("@", 1)
                except ValueError:
                    logging.warning(f"Invalid URL format in {field}.{idx}: {url}")
                    continue

                save_dir = Path(self.resources) / subdir
                save_dir.mkdir(parents=True, exist_ok=True)

                local_path = None

                # === Handle modelscope ===
                if type_ == "modelscope":
                    try:
                        model_id, file_path = uri.split(":", 1)
                        from modelscope.hub.file_download import model_file_download
                        local_path = model_file_download(
                            model_id=model_id,
                            file_path=file_path,
                            local_dir=save_dir
                        )
                    except Exception as e:
                        logging.error(f"Failed to download modelscope model for {field}.{idx}: {e}")
                        continue

                # === Handle http/https ===
                elif type_ in ("http", "https"):
                    try:
                        parsed = urlparse(uri)
                        filename = os.path.basename(parsed.path) or "downloaded_file"
                        save_path = save_dir / filename

                        if not save_path.exists():
                            logging.info(f"Downloading {uri} to {save_path}")
                            r = requests.get(uri, stream=True, timeout=10)
                            r.raise_for_status()
                            with open(save_path, "wb") as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            logging.info(f"Saved to {save_path}")
                        else:
                            logging.info(f"File already exists: {save_path}")

                        local_path = save_path

                    except Exception as e:
                        logging.error(f"Failed to download http resource for {field}.{idx}: {e}")
                        continue
                elif type_.startswith("template"):
                    logging.info(f"Skip processing template resource for {field}.{idx}")
                    continue
                else:
                    logging.warning(f"Unsupported or failed to download resource for {field}.{idx} with type '{type_}'")

                # === Store result ===
                if local_path:
                    result_paths[f"{field}.{idx}"] = str(local_path)

        return result_paths


class Executor:
    """Resident Graph or Python execution, owned entirely by one worker thread."""

    def __init__(self, source, config, storage=None, workdir=None, *, cpu=None):
        self.source, self.config = source, config
        self.storage, self.workdir = storage, workdir
        self.runner = GraphRunner()
        self.binding = None
        self.handler = None
        self.loop = None
        self.cpu = cpu
        self.asset_lease = None
        if config.execution.cpu_threads is not None and cpu is None:
            raise ValueError('CPU thread settings require an isolated Worker process')
        self.is_graph = any(source.get(key) is not None for key in ('workflow', 'graph', 'graph_factory'))

    def open(self, deadline=None):
        import asyncio
        import inspect
        import time
        deadline = min(deadline if deadline is not None else float('inf'),
                       time.monotonic() + self.config.execution.startup_timeout_seconds)
        self._check_startup_deadline(deadline)
        if self.cpu is not None:
            self.cpu.apply()
        from .arrays import check_devices
        check_devices(self.config)
        prepared_resources = None
        if self.is_graph:
            if self.asset_lease is None and self.storage is not None:
                from .assets import AssetLease
                self.asset_lease = AssetLease(self.source.get('resource_dir') or self.workdir, self.storage.db)
            if self.source.get('plugin_dir'):
                from .worker import load_existing_plugins
                plugin_dir = Path(self.source['plugin_dir'])
                if plugin_dir.is_dir():
                    load_existing_plugins(plugin_dir)
            if self.config.execution.batching.enabled:
                raise ValueError('Dynamic batching requires an algorithm handler with deploy_batch and batch_key')
            if self.config.streaming.enabled:
                raise ValueError('Streaming requires a code entry that yields incremental outputs')
            from .io import GraphIO, prepare_native_workflow, prepare_native_graph
            workflow = self.source.get('workflow')
            if workflow is not None:
                from .resources import prepare_workflow
                workflow = prepare_workflow(workflow, self.config, self.source.get('resource_dir') or self.workdir,
                    self.source.get('base_dir') or Path.cwd(), storage=self.storage, holder=self.source.get('resource_holder'),
                    lease=self.asset_lease)
            workflow, native_originals = prepare_native_workflow(workflow, self.config)
            def prepare(graph):
                nonlocal prepared_resources
                if self.source.get('autodetect'):
                    from .config import ServerConfig, suggest_config
                    generated = suggest_config(json.loads(graph.serialize()))
                    generated.update(name=self.config.name, execution=self.config.execution.model_dump(),
                                     mcp=self.config.mcp.model_dump() if self.config.mcp else None)
                    self.config = ServerConfig.model_validate(generated)
                prepare_native_graph(graph, self.config, native_originals)
                if self.source.get('workflow') is None:
                    from .resources import prepare_graph
                    prepared_resources = prepare_graph(
                        graph, self.config, self.source.get('resource_dir') or self.workdir,
                        self.source.get('base_dir') or Path.cwd(),
                        snapshot=self.source.get('resource_snapshot'), fallback=self.source.get('resource_fallback'),
                        storage=self.storage, holder=self.source.get('resource_holder'), lease=self.asset_lease)
                if self.cpu is not None:
                    self.cpu.configure_graph(graph)
            graph = self.runner.load(workflow, graph=self.source.get('graph'),
                                     graph_factory=self.source.get('graph_factory'), before_init=prepare)
            try:
                self.binding = GraphIO(graph, self.config, max_bytes=self.storage.max_bytes, native_originals=native_originals)
            except BaseException:
                self.runner.close()
                raise
        else:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            factory = self.source.get('handler_factory')
            self.handler = factory() if factory else self.source['deploy']
            if self.cpu is not None:
                self.cpu.apply()
            if self.config.execution.batching.enabled:
                for name in ('batch_key', 'deploy_batch'):
                    method = getattr(self.handler, name, None)
                    if not callable(method) or inspect.iscoroutinefunction(method) or inspect.isgeneratorfunction(method) or inspect.isasyncgenfunction(method):
                        raise ValueError('Dynamic batching requires synchronous batch_key and deploy_batch methods')
            initialize = getattr(self.handler, 'init', None)
            if initialize:
                value = initialize()
                if inspect.isawaitable(value):
                    self.loop.run_until_complete(value)
        self._check_startup_deadline(deadline)
        for values in self.warmup_requests():
            self._check_startup_deadline(deadline)
            if self.config.execution.batching.enabled:
                self.batch_key(values)
                result = self.deploy_batch([values])[0]
                if isinstance(result, BaseException):
                    raise result
            else:
                self.deploy(values, emit=lambda value: None)
            self._check_startup_deadline(deadline)

        if prepared_resources is not None:
            self.source['resource_snapshot'] = prepared_resources

    @staticmethod
    def _check_startup_deadline(deadline):
        import time
        from .errors import ServerError
        if time.monotonic() >= deadline:
            raise ServerError(504, 'startup_timeout', 'Worker initialization and warmup exceeded the startup deadline')

    def warmup_requests(self):
        import uuid
        from .config import validate_request
        for sample in self.config.execution.warmup:
            values = validate_request(self.config, sample.inputs, sample.parameters,
                                      input_model=self.source.get('input_model'), untyped=self.source.get('untyped', False))
            for group in ('inputs', 'parameters'):
                for name, field in getattr(self.config, group).items():
                    if field.is_file():
                        _, record = self.storage.resolve_input(values[group][name], field)
                        field.check_file(record)
            yield dict(values, _task_id='warmup_' + uuid.uuid4().hex, _warmup=True)

    def deploy(self, values, emit=None):
        import asyncio
        from .errors import ServerError
        from .logging_taskid import set_task_id, reset_task_id, set_request_id, reset_request_id
        token = set_task_id(values.get('_task_id', '0'))
        request_token = set_request_id(values.get('_request_id'))
        try:
            return self._deploy(values, emit)
        except asyncio.CancelledError as exc:
            # This boundary runs on the owning compute thread/process. The
            # algorithm has unwound; HTTP waiter cancellation never enters it.
            # Do not leak a BaseException into the queue consumer or terminate
            # a reusable process Worker for one cancelled algorithm request.
            raise ServerError(409, 'task_cancelled', 'Algorithm request stopped',
                              task_id=values.get('_task_id')) from exc
        finally:
            reset_request_id(request_token)
            reset_task_id(token)

    def _deploy(self, values, emit=None):
        import copy
        import inspect
        import tempfile
        if self.cpu is not None:
            self.cpu.apply()
        if not self.is_graph:
            target = getattr(self.handler, 'deploy', self.handler)
            inputs, parameters = self.decode_inputs(values)
            if self.config.streaming.duplex:
                parameters['input_stream'] = self.input_stream(values)
            result = target(inputs, **parameters)
            if inspect.isawaitable(result):
                result = self.loop.run_until_complete(result)
            if inspect.isgenerator(result) or inspect.isasyncgen(result):
                if not self.config.streaming.enabled or emit is None:
                    if inspect.isasyncgen(result):
                        self.loop.run_until_complete(result.aclose())
                    else:
                        result.close()
                    raise ValueError('Generator entries must explicitly enable streaming')
                result = self._consume_stream(result, emit, values)
            elif self.config.streaming.enabled:
                raise ValueError('Streaming entries must return a generator')
            return self.validate_result(result, values)
        if self.runner.graph is None:
            self.open()
        input_paths = {}
        for group in ('inputs', 'parameters'):
            for name, field in getattr(self.config, group).items():
                if field.is_file():
                    path, record = self.storage.resolve_input(values[group][name], field)
                    field.check_file(record)
                    input_paths[group, name] = path
        published = []
        try:
            with tempfile.TemporaryDirectory(prefix='request-', dir=self.workdir) as directory:
                from .storage.media import MEDIA_EXTENSIONS
                paths = {name: Path(directory) / f'output-{index}{MEDIA_EXTENSIONS[field.output_media_type()]}'
                         for index, (name, field) in enumerate(self.config.outputs.items())
                         if field.is_file()}
                result = self.runner.execute(
                    lambda graph: self.binding.bind(graph, values, input_paths, paths,
                        before_init=self.cpu.configure_graph if self.cpu is not None else None),
                    lambda graph: self.binding.collect(graph, paths),
                    steps=lambda: self.binding.steps,
                    after_step=lambda: self.binding.check_output_sizes(paths))
                if values.get('_warmup'):
                    from .storage.media import validate
                    from .errors import ServerError
                    for name in paths:
                        if result[name].stat().st_size > self.storage.max_bytes:
                            raise ServerError(413, 'file_too_large', 'Warmup output exceeds the upload limit')
                        validate(result[name], self.config.outputs[name].output_media_type())
                    # The temporary directory owns warmup output. Nothing is
                    # published to storage, tasks or the event history.
                    return None
                for name in paths:
                    with result[name].open('rb') as stream:
                        record = self.storage.save(stream, self.config.outputs[name].output_media_type(),
                                                   owner=values.get('_owner'), task_id=values['_task_id'],
                                                   attempt=values.get('_attempt', 1), worker_token=values.get('_worker_token'))
                        result[name] = {'file_id': record['file_id']}
                    published.append(result[name]['file_id'])
                return result
        except BaseException:
            for identity in published:
                try:
                    self.storage.discard_output(identity, values['_task_id'], attempt=values.get('_attempt', 1),
                                                worker_token=values.get('_worker_token'))
                except Exception:
                    logging.exception('Could not clean failed output %s', identity)
            raise

    def batch_key(self, values):
        from .logging_taskid import set_task_id, reset_task_id, set_request_id, reset_request_id
        task_token = set_task_id(values.get('_task_id', '0'))
        request_token = set_request_id(values.get('_request_id'))
        try:
            return self._batch_key(values)
        finally:
            reset_request_id(request_token)
            reset_task_id(task_token)

    def _batch_key(self, values):
        from .db import encode
        inputs, parameters = self.decode_inputs(values)
        key = self._batch_call(self.handler.batch_key, inputs, **parameters)
        encoded = encode([values.get('_owner'), values['parameters'], key])
        if len(encoded.encode('utf-8')) > 4096:
            raise ValueError('Batch compatibility key exceeds 4096 bytes')
        return encoded

    def _batch_call(self, method, *args, **kwargs):
        import asyncio
        import inspect
        from .errors import ServerError
        try:
            result = method(*args, **kwargs)
        except asyncio.CancelledError as exc:
            # Batch methods run on the compute thread, just like _deploy.
            # One algorithm cancellation must not cancel the batch scheduler.
            raise ServerError(409, 'task_cancelled', 'Algorithm batch request stopped') from exc
        if inspect.isasyncgen(result):
            self.loop.run_until_complete(result.aclose())
        elif inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
        elif inspect.isgenerator(result):
            result.close()
        else:
            return result
        raise ValueError('Dynamic batch methods must return synchronous values')

    def deploy_batch(self, values):
        import copy
        from .errors import ServerError
        from .logging_taskid import set_task_id, reset_task_id, set_request_id, reset_request_id
        # No single task owns a batch. Algorithms receive each task ID explicitly.
        context = set_task_id('0')
        request_context = set_request_id(None)
        try:
            items = []
            for value in values:
                inputs, parameters = self.decode_inputs(value)
                items.append(dict(task_id=value['_task_id'], request_id=value.get('_request_id'), inputs=inputs, parameters=parameters))
            results = self._batch_call(self.handler.deploy_batch, items)
            if not isinstance(results, list) or len(results) != len(items):
                raise ServerError(500, 'batch_result_mismatch', 'Algorithm must return one result for each batch item')
            validated = []
            for result, value in zip(results, values):
                try:
                    validated.append(result if isinstance(result, ServerError) else self.validate_result(result, value))
                except Exception:
                    validated.append(ServerError(500, 'invalid_result', 'Batch item does not match the output schema'))
            return validated
        finally:
            reset_request_id(request_context)
            reset_task_id(context)

    def decode_inputs(self, values):
        import copy
        from .arrays import decode
        groups = []
        for group in ('inputs', 'parameters'):
            value = copy.deepcopy(values[group])
            for name, field in getattr(self.config, group).items():
                if field.type in ('array', 'tensor'):
                    value[name] = decode(field, value[name], storage=self.storage)
            groups.append(value)
        return groups

    def _prepare_output(self, result):
        import copy
        import io
        from .arrays import encode
        files = {}
        try:
            if isinstance(result, dict):
                result = dict(result)
                for name, field in self.config.outputs.items():
                    if name in result and field.type in ('array', 'tensor'):
                        if field.is_file() and not (isinstance(result[name], dict) and set(result[name]) == {'file_id'}):
                            stream = files[name] = io.BytesIO()
                            encode(field, result[name], stream=stream, snapshot=True)
                            if not field.is_audio() and stream.tell() > self.storage.max_bytes:
                                raise ValueError('Array output exceeds the file byte limit')
                            stream.seek(0)
                            result[name] = None
                        else:
                            result[name] = encode(field, result[name])
                if files:
                    result = copy.deepcopy(result)
            return result, files
        except BaseException:
            for stream in files.values():
                stream.close()
            raise

    def _save_array_files(self, result, files, values):
        import io
        try:
            for name in files:
                stream = files[name]
                field = self.config.outputs[name]
                if field.is_audio():
                    from .arrays import read_npy
                    from .audio import write_audio
                    array = read_npy(stream, field)
                    stream.close()
                    stream = files[name] = io.BytesIO()
                    write_audio(stream, array, field, max_bytes=self.storage.max_bytes)
                    stream.seek(0)
                if values.get('_warmup'):
                    result[name] = {'file_id': 'warmup'}
                else:
                    record = self.storage.save(stream, field.media_type,
                        owner=values.get('_owner'), task_id=values['_task_id'],
                        attempt=values.get('_attempt', 1), worker_token=values.get('_worker_token'))
                    result[name] = {'file_id': record['file_id']}
            return result
        finally:
            for stream in files.values():
                stream.close()

    def encode_output(self, result, values=None):
        result, files = self._prepare_output(result)
        return self._save_array_files(result, files, values)

    async def encode_output_async(self, result, values):
        # Capture every numeric buffer on the algorithm's loop before the first
        # await. Another engine request may reuse the same buffer while storage
        # is running on a thread. Only owned binary data crosses that await.
        result, files = self._prepare_output(result)
        if not files:
            return result
        return await self.array_file_io(self._save_array_files, result, files, values)

    def validate_result(self, result, values=None):
        import copy
        result = self.encode_output(result, values)
        model = self.source.get('output_model')
        if model is not None:
            result = model.model_validate(result, strict=True).model_dump(mode='json')
        if self.config.outputs:
            if not isinstance(result, dict) or result.keys() != self.config.outputs.keys():
                raise ValueError('Result fields do not match the declared outputs')
            result = {name: field.validate_value(result[name]) for name, field in self.config.outputs.items()}
        json.dumps(result, allow_nan=False)
        return copy.deepcopy(result)

    async def open_async(self):
        """Initialize one engine on the event loop which owns all its requests."""
        import inspect
        from .arrays import check_devices
        check_devices(self.config)
        if self.is_graph:
            raise ValueError('Graph instances require serial execution; engine mode is for async code')
        factory = self.source.get('handler_factory')
        self.handler = factory() if factory else self.source['deploy']
        if inspect.isawaitable(self.handler):
            self.handler = await self.handler
        target = getattr(self.handler, 'deploy', self.handler)
        declaration = target if inspect.isfunction(target) or inspect.ismethod(target) else getattr(target, '__call__', target)
        asynchronous = inspect.iscoroutinefunction(declaration) or inspect.isasyncgenfunction(declaration)
        if self.config.execution.scheduling == 'concurrent' and asynchronous:
            raise ValueError('Concurrent scheduling requires synchronous thread-safe code; async code uses engine scheduling')
        if self.config.execution.scheduling == 'engine' and not asynchronous:
            raise ValueError('Engine deploy must be an async function or async generator')
        if (inspect.isasyncgenfunction(declaration) or inspect.isgeneratorfunction(declaration)) and not self.config.streaming.enabled:
            raise ValueError('Generator entries must explicitly enable streaming')
        initialize = getattr(self.handler, 'init', None)
        if initialize:
            result = initialize()
            if inspect.isawaitable(result):
                await result

    async def deploy_async(self, values, publish):
        import copy
        import inspect
        target = getattr(self.handler, 'deploy', self.handler)
        inputs, parameters = await self.array_file_io(self.decode_inputs, values)
        if self.config.streaming.duplex:
            parameters['input_stream'] = self.input_stream(values)
        result = target(inputs, **parameters)
        if inspect.isawaitable(result):
            result = await result
        if self.config.streaming.enabled:
            if not inspect.isasyncgen(result):
                raise ValueError('Streaming engines must return an async generator')
            iterator, result = result, None
            try:
                async for value in iterator:
                    result = await publish(value)
            finally:
                await iterator.aclose()
        elif inspect.isasyncgen(result):
            await result.aclose()
            raise ValueError('Generator entries must explicitly enable streaming')
        return self.validate_result(await self.encode_output_async(result, values), values)

    async def array_file_io(self, operation, *args):
        """Keep numeric file I/O off an engine loop and join it on cancellation."""
        import asyncio
        if not any(field.type in ('array', 'tensor') and field.is_file()
                   for group in (self.config.inputs, self.config.parameters, self.config.outputs)
                   for field in group.values()):
            return operation(*args)
        pending = asyncio.create_task(asyncio.to_thread(operation, *args))
        interrupted = None
        while not pending.done():
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError as exc:
                interrupted = exc
            except Exception:
                break
        if interrupted is not None:
            if not pending.cancelled():
                pending.exception()
            raise interrupted
        return pending.result()

    async def input_stream(self, values):
        import asyncio
        import time
        from .errors import ServerError
        if values.get('_warmup'):
            return
        after = 0
        deadline = time.monotonic() + self.config.streaming.max_duration_seconds
        while True:
            if time.monotonic() >= deadline:
                raise ServerError(504, 'execution_timeout', 'Duplex input exceeded its duration')
            reading = asyncio.create_task(asyncio.to_thread(self.storage.db.read_task_input, values['_task_id'], after,
                attempt=values.get('_attempt', 1), worker_token=values.get('_worker_token')))
            try:
                value = await asyncio.shield(reading)
            except asyncio.CancelledError:
                # A cancelled reader must not leave database work outliving
                # its Worker and connection pool.
                await reading
                raise
            if value is None:
                await asyncio.sleep(.05)
                continue
            if value['end']:
                return
            yield value['data']
            after = value['sequence']

    async def close_async(self):
        import inspect
        if self.handler is not None:
            try:
                close = getattr(self.handler, 'deinit', None)
                if close:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
            finally:
                self.handler = None

    def _consume_stream(self, iterator, emit, values):
        import asyncio
        import inspect
        import time
        from .db import encode
        from .errors import ServerError
        deadline = time.monotonic() + self.config.streaming.max_duration_seconds
        last = None
        count = 0
        def publish(value):
            nonlocal last, count
            if time.monotonic() >= deadline:
                raise ServerError(504, 'execution_timeout', 'Stream exceeded its execution duration')
            count += 1
            if count > self.config.streaming.max_events:
                raise ServerError(422, 'stream_event_limit', 'Stream event count exceeds the configured limit')
            serialized = encode(self.encode_output(value, values))
            if len(serialized.encode('utf-8')) > self.config.streaming.max_event_bytes:
                raise ServerError(422, 'stream_event_too_large', 'Stream event exceeds the configured limit')
            last = json.loads(serialized)
            emit(last)
        if inspect.isasyncgen(iterator):
            async def consume():
                try:
                    async for value in iterator:
                        publish(value)
                finally:
                    await iterator.aclose()
            try:
                self.loop.run_until_complete(asyncio.wait_for(consume(), self.config.streaming.max_duration_seconds))
            except asyncio.TimeoutError as exc:
                raise ServerError(504, 'execution_timeout', 'Stream exceeded its execution duration') from exc
        else:
            try:
                for value in iterator:
                    publish(value)
            finally:
                iterator.close()
        return last

    def close(self):
        import asyncio
        import inspect
        try:
            if self.is_graph:
                self.binding = None
                self.runner.close()
                if self.asset_lease is not None:
                    self.asset_lease.close()
                    self.asset_lease = None
            elif self.handler is not None:
                close = getattr(self.handler, 'deinit', None)
                if close:
                    result = close()
                    if inspect.isawaitable(result):
                        self.loop.run_until_complete(result)
        finally:
            self.handler = None
            if self.loop is not None:
                try:
                    pending = asyncio.all_tasks(self.loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                    self.loop.run_until_complete(self.loop.shutdown_default_executor())
                finally:
                    self.loop.close()
                    self.loop = None
                    asyncio.set_event_loop(None)
