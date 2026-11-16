"""Persist editor output previews and files using the existing task and storage."""
import math

MAX_OUTPUT_BYTES = 1024 * 1024
MAX_NODE_BYTES = 64 * 1024
MAX_NODES = 256
MAX_FILES = 256
MAX_VALUES = 1024


def memory_outputs(results):
    """Do not stringify native objects or persist Python serialization."""
    content, truncated, remaining = {}, False, MAX_OUTPUT_BYTES
    if not isinstance(results, dict):
        return content, False
    for index, (name, outputs) in enumerate(results.items()):
        if index >= MAX_NODES or remaining <= 0:
            truncated = True
            break
        if not isinstance(name, str) or not isinstance(outputs, dict):
            continue
        name_bytes = name[:1025].encode('utf-8', errors='replace')
        if len(name_bytes) > 1024 or len(name_bytes) > remaining:
            truncated = True
            continue
        budget = min(MAX_NODE_BYTES, remaining - len(name_bytes))
        parts, used, count = [], 0, 0
        for kind in ('String', 'Bool', 'Num'):
            if kind not in outputs:
                continue
            values = outputs[kind] if isinstance(outputs[kind], list) else [outputs[kind]]
            for value in values:
                count += 1
                if count > MAX_VALUES or used >= budget:
                    truncated = True
                    break
                if type(value) not in (str, bool, int, float, type(None)):
                    continue
                if type(value) is float and not math.isfinite(value):
                    continue
                if type(value) is int and value.bit_length() > 4096:
                    truncated = True
                    continue
                text = value if type(value) is str else str(value)
                space = max(0, budget - used - bool(parts))
                data = text[:space].encode('utf-8', errors='replace')
                clipped = data[:space].decode('utf-8', errors='ignore')
                truncated |= len(text) > space or len(data) > space
                used += len(clipped.encode('utf-8')) + bool(parts)
                parts.append(clipped)
        if parts:
            content[name] = ' '.join(parts)
            remaining -= len(name_bytes) + used
    return content, truncated


class EditorResults(dict):
    """Carry safe previews even when another output cannot cross Worker IPC."""
    def __init__(self, values, preview):
        super().__init__(values)
        self.preview = preview


def preview(results):
    return results.preview if isinstance(results, EditorResults) else memory_outputs(results)


def file_outputs(workflow):
    """Locate declared output files without changing workflow JSON or nodes."""
    known = {'nndeploy.codec.PILImageEncodec', 'nndeploy::codec::OpenCvImageEncode',
             'nndeploy::codec::OpenCvVideoEncode'}
    result = []
    seen = set()
    visited = 0
    def add(path, field_path, value, depth=0):
        nonlocal visited
        key = (tuple(path), tuple(field_path))
        if key in seen:
            return
        seen.add(key)
        visited += 1
        if visited > MAX_VALUES or depth > 64:
            raise ValueError('File output lists exceed the supported limit')
        if isinstance(value, list):
            if len(value) > MAX_FILES:
                raise ValueError('Too many output files')
            for index, item in enumerate(value):
                add(path, field_path + [index], item, depth + 1)
            return
        if value is None or value == '':
            return
        if not isinstance(value, str):
            raise ValueError('File output paths must be strings or lists of paths')
        result.append((path, field_path, value))
        if len(result) > MAX_FILES:
            raise ValueError('Too many output files')

    def visit(nodes, parents):
        if len(parents) > 64:
            raise ValueError('Output node nesting exceeds the supported limit')
        for node in nodes:
            path = parents + [node['name_']]
            if node.get('key_') in known:
                add(path, ['path_'], node.get('path_'))
            elif node.get('key_') == 'nndeploy.basic.ConsolePrintNode':
                settings = (node.get('frontend_params') or {})
                if settings.get('save_to_file'):
                    add(path, ['frontend_params', 'file_path'], settings.get('file_path'))
            elif node.get('node_type_') == 'Output' and node.get('io_type_') in ('Image', 'Video', 'Audio', 'File', 'Binary', 'Text', 'Dir'):
                for prefix, fields in (([], node), (['param_'], node.get('param_') or {}), (['param'], node.get('param') or {})):
                    names = next((fields[key] for key in ('io_param_', 'io_params_', 'required_params_') if fields.get(key)), [])
                    if not isinstance(names, list) or len(names) > MAX_FILES:
                        raise ValueError('File output fields must be a bounded list')
                    for name in names:
                        if not isinstance(name, str) or not name or len(name.encode('utf-8')) > 1024:
                            raise ValueError('File output field names must be nonempty bounded strings')
                        add(path, prefix + [name], fields.get(name))
            visit(node.get('node_repository_', []), path)
    visit(workflow.get('node_repository_', []), [])
    return result


def capture_files(workflow, storage, scope, task_id, token):
    """Copy files before this Worker starts a later task using the same paths."""
    import mimetypes
    import os
    import stat
    saved = storage.db.editor_output_files(scope, task_id)
    existing = {(tuple(item['node_path']), tuple(item['field_path'])) for item in saved if item.get('field_path') is not None}
    legacy = {tuple(item['node_path']) for item in saved if item.get('field_path') is None}
    validated = {'image/png', 'image/jpeg', 'image/webp', 'text/plain', 'application/json', 'application/zip',
                 'audio/wav', 'audio/mpeg', 'audio/ogg', 'audio/flac', 'audio/mp4', 'video/mp4', 'video/webm'}
    for node_path, field_path, source in file_outputs(workflow):
        if tuple(node_path) in legacy:
            # Older Workers captured only the first nonempty field per node.
            # Keep its original file; do not re-read a mutable output path.
            legacy.remove(tuple(node_path))
            continue
        if (tuple(node_path), tuple(field_path)) in existing:
            continue  # A lost receipt must not recopy a subsequently changed file.
        media_type = mimetypes.guess_type(source)[0]
        if media_type not in validated:
            media_type = 'application/octet-stream'
        # O_NONBLOCK prevents a declared FIFO from hanging before fstat can
        # reject it. Inspect the opened descriptor, not a raceable path stat.
        descriptor = os.open(source, os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0))
        try:
            mode = os.fstat(descriptor).st_mode
            options = dict(owner='editor:' + scope, editor_task=(scope, task_id, token, node_path, field_path))
            if stat.S_ISDIR(mode):
                from .storage.archive import directory_archive
                with directory_archive(descriptor, max_bytes=storage.max_bytes, temp_dir=storage.root) as content:
                    storage.save(content, 'application/zip', **options)
            elif stat.S_ISREG(mode):
                with os.fdopen(os.dup(descriptor), 'rb') as content:
                    storage.save(content, media_type, **options)
            else:
                raise ValueError('Output files must be regular files or directories')
        finally:
            os.close(descriptor)
    return storage.db.editor_output_files(scope, task_id)


def capture_worker_files(payload, options):
    if options is None or not file_outputs(payload['graph_json']):
        return
    from .db import DB
    from .storage import LocalStorage
    db = DB(options['database_url'], timeout_seconds=options['timeout_seconds'])
    storage = None
    try:
        storage = LocalStorage(options['root'], db, max_bytes=options['max_bytes'], config=options['config'])
        capture_files(payload['graph_json'], storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
    finally:
        if storage is not None:
            storage.close()
        db.close()
