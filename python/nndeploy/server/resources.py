"""Keep a running workflow independent from its editable JSON and local assets."""
import copy
import hashlib
import os
import shutil
import tempfile
import uuid
from pathlib import Path


# File-valued native parameters. HTTP callers cannot supply arbitrary paths.
_FILE_FIELDS = {'path_', 'model_path_', 'model_value_', 'tokenizer_path_', 'vocab_file_',
                'merges_file_', 'config_path_', 'weights_path_', 'checkpoint_path_'}
_RESOURCE_TYPES = {'Model', 'Dir', 'Binary', 'File', 'Text', 'Json', 'Xml', 'Csv', 'Yaml',
                   'Image', 'Video', 'Audio'}
_OUTPUT_CODECS = {'nndeploy.codec.PILImageEncodec', 'nndeploy::codec::OpenCvImageEncode',
                  'nndeploy::codec::OpenCvImagesEncode', 'nndeploy::codec::OpenCvVideoEncode'}


def server_scope(directory):
    """Stable private workspace identity across process and container restarts."""
    directory = Path(directory)
    target = directory / '.server-id'
    if not target.exists():
        pending = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=directory, prefix='.scope-', delete=False) as output:
                pending = Path(output.name)
                output.write(uuid.uuid4().hex)
                output.flush()
                os.fsync(output.fileno())
            try:
                # Publish a complete file exclusively; concurrent starters must
                # not read another process's partly written identity.
                os.link(pending, target)
            except FileExistsError:
                pass
        finally:
            if pending:
                pending.unlink(missing_ok=True)
    value = target.read_text().strip()
    if uuid.UUID(value).hex != value:
        raise ValueError('Invalid Server workspace identity')
    if os.name != 'nt':
        # Commit the directory entry before a remote DB can save configurations
        # under this identity. A file fsync alone does not persist its new name.
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return value


def _digest(path):
    digest = hashlib.sha256()
    paths = sorted(path.rglob('*')) if path.is_dir() else [path]
    for item in paths:
        if item.is_symlink():
            raise ValueError('Managed resource directories cannot contain symbolic links')
        if not item.is_file() and not item.is_dir():
            raise ValueError('Managed resources must be regular files or directories')
        name = str(item.relative_to(path) if path.is_dir() else item.name).encode()
        digest.update(len(name).to_bytes(8, 'big'))
        digest.update(name)
        digest.update(b'd' if item.is_dir() else b'f')
        if item.is_file():
            digest.update(item.stat().st_size.to_bytes(8, 'big'))
            with item.open('rb') as content:
                while chunk := content.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


def _verify_asset(path, assets, verified=None):
    """Validate the complete copied resource, including a referenced subpath."""
    path = Path(path)
    if not path.is_absolute() or not path.resolve().is_relative_to(assets):
        raise ValueError('Saved Graph resources must remain inside the Server assets directory')
    parts = path.relative_to(assets).parts
    if len(parts) < 2 or not path.exists():
        raise ValueError('A saved Graph resource no longer exists')
    root = assets / parts[0] / parts[1]
    if verified is None or root not in verified:
        if root.is_symlink() or _digest(root) != parts[0]:
            raise ValueError('Saved Graph resource integrity check failed')
        if verified is not None:
            verified.add(root)
    return str(path)


def _file_reference(value):
    return (isinstance(value, dict) and set(value) == {'file_id'} and
            isinstance(value['file_id'], str) and value['file_id'].startswith('file_') and len(value['file_id']) <= 64)


def _map_resources(workflow, transform, bound_nodes=(), _node_path=()):
    """Use one set of Graph resource declarations for references and loading."""
    def visit(node, node_path=(), io_type=None, output=False, field_path=()):
        if len(node_path) > 64:
            raise ValueError('Resource node nesting exceeds the supported limit')
        io_type = node.get('io_type_', io_type)
        output = output or node.get('node_type_') in ('Output', 'kNodeTypeOutput') or node.get('key_') in _OUTPUT_CODECS
        declared = set()
        if io_type in _RESOURCE_TYPES and node_path not in bound_nodes and not output:
            names = node.get('io_param_') or node.get('io_params_') or []
            if not isinstance(names, list) or len(names) > 1024:
                raise ValueError('Resource I/O fields must be a bounded list')
            for name in names:
                if not isinstance(name, str) or not name or len(name.encode('utf-8')) > 1024 or name not in node:
                    raise ValueError('Resource I/O fields must name existing fields')
                declared.add(name)
        for key, value in list(node.items()):
            if ((key in _FILE_FIELDS or key in declared) and not output and
                    node_path not in bound_nodes and not (key == 'model_value_' and node.get('is_path_') is False)):
                node[key] = transform(value, declared=key in declared, directory=io_type == 'Dir',
                                      node_path=node_path, field_path=(*field_path, key))
            elif key == 'node_repository_':
                for child in value:
                    visit(child, (*node_path, child['name_']))
            elif isinstance(value, dict):
                visit(value, node_path, io_type, output, (*field_path, key))
        return node
    return visit(copy.deepcopy(workflow), _node_path)


def workflow_file_refs(workflow):
    """Return only managed files in static input/resource fields, not previews."""
    references = []
    def collect(value, *, node_path, field_path, depth=0, **_):
        if depth > 64:
            raise ValueError('Resource path lists exceed the supported nesting limit')
        if isinstance(value, list):
            if len(value) > 1024:
                raise ValueError('Resource path lists exceed the supported item limit')
            for index, item in enumerate(value):
                collect(item, node_path=node_path, field_path=(*field_path, index), depth=depth + 1)
        elif _file_reference(value):
            references.append((node_path, field_path, value['file_id']))
        return value
    _map_resources(workflow, collect)
    return references


def prepare_workflow(workflow, config, directory, base_dir, *, _node_path=(), storage=None, holder=None, _copy_local=True, lease=None):
    """Copy referenced local assets and return a separate runtime JSON object."""
    assets = Path(directory).resolve() / 'assets'
    if assets.is_symlink():
        raise ValueError('The Server assets directory cannot be a symbolic link')
    assets.mkdir(parents=True, exist_ok=True)
    verified = set()
    bound_nodes = {tuple(field.node) for group in (config.inputs, config.parameters, config.outputs)
                   for field in group.values() if field.node is not None
                   and not (group is config.parameters and field.type == 'json')}

    def publish(source):
        checksum = _digest(source)
        destination_dir = assets / checksum
        if lease is not None:
            lease.hold(destination_dir, created=not destination_dir.exists())
        destination_dir.mkdir(exist_ok=True)
        if destination_dir.is_symlink() or destination_dir.resolve().parent != assets:
            raise ValueError('Managed resource directories cannot redirect outside Server assets')
        destination = destination_dir / source.name
        if not destination.exists():
            try:
                source.rename(destination)
            except OSError:
                if not destination.exists():
                    raise
        return _verify_asset(destination, assets, verified)

    def materialize(reference, directory):
        if storage is None:
            # Saved workflow JSON retains the file ID. Only its Worker copy
            # resolves paths, after the existing runtime reference is held.
            return copy.deepcopy(reference)
        from .errors import ServerError
        from .storage.media import MEDIA_EXTENSIONS
        identity = reference['file_id']
        if holder is not None:
            storage.db.hold_files(holder, reference)
        with storage.open(identity) as stream:
            record = storage.metadata(identity)
            if directory and record['media_type'] != 'application/zip':
                raise ValueError('Directory model resources require a ZIP file')
            with (lease.temporary('.materialize-') if lease is not None else
                  tempfile.TemporaryDirectory(prefix='.materialize-', dir=assets)) as pending:
                source = Path(pending) / (identity + MEDIA_EXTENSIONS.get(record['media_type'], '.bin'))
                with source.open('wb') as target:
                    size, checksum = storage._copy(stream, target, min(storage.max_bytes, record['size']))
                if size != record['size'] or checksum != record.get('sha256'):
                    raise ServerError(502, 'content_mismatch', 'Model resource failed integrity validation')
                if directory:
                    from .storage.archive import extract_archive, MAX_BYTES
                    expanded = Path(pending) / identity
                    extract_archive(source, expanded, max_bytes=max(MAX_BYTES, storage.max_bytes))
                    source = expanded
                return publish(source)

    def preserve(value, *, declared=False, depth=0, directory=False, **_):
        if depth > 64:
            raise ValueError('Resource path lists exceed the supported nesting limit')
        if isinstance(value, list):
            if len(value) > 1024:
                raise ValueError('Resource path lists exceed the supported item limit')
            return [preserve(item, declared=declared, depth=depth + 1, directory=directory) for item in value]
        if _file_reference(value):
            return materialize(value, directory)
        if not isinstance(value, str) or not value or '://' in value:
            if declared and value not in ('', None):
                raise ValueError('Declared resources require local paths or file_id references')
            return value
        if not _copy_local:
            return value  # Legacy editor paths keep their original Run semantics.
        original = Path(value).expanduser()
        candidates = [original] if original.is_absolute() else [Path(base_dir) / original, Path.cwd() / original]
        if declared and not original.is_absolute():
            candidates = [Path(base_dir) / original]
        for candidate in candidates:
            lexical = Path(os.path.abspath(candidate))
            if lexical.is_relative_to(assets):
                if lease is not None:
                    lease.hold(lexical)
                return _verify_asset(lexical, assets, verified)
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            if declared:
                raise ValueError('A declared resource file or directory does not exist')
            # Native parameters may contain backend names rather than filenames.
            return value
        if path.is_symlink():
            raise ValueError('Managed resources must be regular files or directories')
        if not path.is_file() and not path.is_dir():
            raise ValueError('Managed resources must be regular files or directories')
        lexical = Path(os.path.abspath(path))
        if lexical.is_relative_to(assets):
            if lease is not None:
                lease.hold(lexical)
            return _verify_asset(lexical, assets, verified)
        path = path.resolve()
        if path.is_relative_to(assets):
            if lease is not None:
                lease.hold(path)
            return _verify_asset(path, assets, verified)
        with (lease.temporary('.copy-') if lease is not None else
              tempfile.TemporaryDirectory(prefix='.copy-', dir=assets)) as pending:
            source = Path(pending) / path.name
            if path.is_dir():
                # Check before copying so symlinks cannot read outside the resource tree.
                _digest(path)
                shutil.copytree(path, source, symlinks=True)
            else:
                shutil.copy2(path, source, follow_symlinks=False)
            return publish(source)

    return _map_resources(workflow, preserve, bound_nodes, _node_path)


def prepare_graph(graph, config, directory, base_dir, *, snapshot=None, fallback=None, storage=None, holder=None, lease=None):
    """Read existing Node metadata; code Graphs need not support JSON export."""
    import json
    from nndeploy import _nndeploy_internal as native
    from nndeploy.dag.base import io_type_to_name
    from .io import _check_status
    bound = {tuple(field.node) for group in (config.inputs, config.parameters, config.outputs)
             for field in group.values() if field.node is not None
             and not (group is config.parameters and field.type == 'json')}
    visited = {}
    prepared = []
    assets = Path(directory).resolve() / 'assets'
    previous = snapshot if snapshot is not None else fallback or []
    verified = set()

    def retained(original, saved, saved_base):
        if original == saved:
            return original
        if isinstance(original, dict) and isinstance(saved, dict) and original.keys() == saved.keys():
            return {key: retained(value, saved[key], saved_base) for key, value in original.items()}
        if isinstance(original, list) and isinstance(saved, list) and len(original) == len(saved):
            return [retained(value, other, saved_base) for value, other in zip(original, saved)]
        if _file_reference(original) and isinstance(saved, str):
            if storage is None:
                raise ValueError('Managed model resources require Server storage')
            storage.metadata(original['file_id'])
            if lease is not None:
                lease.hold(saved)
            return _verify_asset(saved, assets, verified)
        if isinstance(original, str) and isinstance(saved, str):
            destination = Path(saved)
            if not destination.is_absolute() or not destination.resolve().is_relative_to(assets):
                raise ValueError('Saved Graph resources must remain inside the Server assets directory')
            source = Path(original).expanduser()
            if not source.is_absolute():
                if saved_base != str(Path(base_dir).resolve()):
                    if snapshot is not None:
                        raise ValueError('Graph resource base directory changed while rebuilding a Worker')
                    return original
                source = Path(base_dir) / source
            # An existing registration pins its loaded resources. A newly
            # registered factory can adopt updated source files; only missing
            # sources fall back to that entry's previously saved copies.
            if snapshot is not None or not source.exists():
                if lease is not None:
                    lease.hold(saved)
                return _verify_asset(saved, assets, verified)
            return original
        raise ValueError('Saved Graph resource declarations do not match the factory')

    def fields(target, io_type):
        names = list(target.get_io_params()) if io_type in _RESOURCE_TYPES else []
        result = {'io_params_': names}
        serialized = None
        for key in sorted(_FILE_FIELDS | set(names) | {'is_path_'}):
            if hasattr(target, key):
                result[key] = copy.deepcopy(getattr(target, key))
            elif key in names:
                getter = getattr(target, 'get_' + key.rstrip('_'), None)
                if callable(getter):
                    result[key] = copy.deepcopy(getter())
                else:
                    # Native codecs expose some paths only through their own
                    # serialization. Never serialize/rebuild the owning Graph.
                    if isinstance(target, native.dag.Graph):
                        raise ValueError('Graph resource fields require readable attributes or getters')
                    if serialized is None:
                        serialized = json.loads(target.serialize())
                    if key in serialized:
                        result[key] = serialized[key]
        return result

    def assign(target, key, value):
        if hasattr(target, key):
            setattr(target, key, copy.deepcopy(value))
        else:
            setter = getattr(target, 'set_' + key.rstrip('_'), None)
            if not callable(setter):
                raise ValueError('Resource field requires a writable attribute or setter: ' + key)
            status = setter(copy.deepcopy(value))
            if status is not None:
                _check_status(status)

    def visit(node, path=()):
        if id(node) in visited:
            return
        visited[id(node)] = node
        if len(path) > 64:
            raise ValueError('Resource node nesting exceeds the supported limit')
        if path not in bound and node.get_node_type() != native.dag.NodeType.Output and node.get_key() not in _OUTPUT_CODECS:
            io_type = io_type_to_name.get(node.get_io_type())
            before = dict(fields(node, io_type), io_type_=io_type)
            param = node.get_param()
            if param is not None:
                before['param_'] = fields(param, io_type)
            if storage is not None and holder is not None:
                storage.db.hold_files(holder, before)
            identity = [list(path), node.get_key(), type(node).__module__, type(node).__qualname__]
            saved = next((value for value in previous if value['node'] == identity), None)
            restored = before
            if saved is not None:
                if saved['before'] == before:
                    restored = retained(before, saved['after'], saved.get('base_dir'))
                elif snapshot is not None:
                    raise ValueError('Graph resource declarations changed while rebuilding a Worker')
            after = prepare_workflow(restored, config, directory, base_dir, _node_path=path,
                                     storage=storage, holder=holder, lease=lease)
            if before != after:
                prepared.append(dict(node=identity, base_dir=str(Path(base_dir).resolve()), before=before, after=after))
            for key, value in after.items():
                if value == before.get(key):
                    continue
                if key == 'param_':
                    for name, part in value.items():
                        if part != before[key].get(name):
                            assign(param, name, part)
                else:
                    assign(node, key, value)
        if isinstance(node, native.dag.Graph):
            for child in node.get_nodes():
                visit(child, (*path, child.get_name()))
    visit(graph)
    return prepared


def save_entry(db, scope, name, source, config, enabled, **options):
    if source.get('workflow') is None:
        if source.get('graph_factory') is None or source.get('resource_snapshot') is None:
            return
        # Metadata only: restoration still requires explicit trusted code
        # registration. No Python import path or callable is stored in the DB.
        data = dict(kind='graph_factory', config={'name': name}, resources=source['resource_snapshot'])
        saved = db.get_server_config(scope, name)
        if saved is not None and saved['data'] == data:
            return saved['write_token']
    else:
        data = dict(workflow=source['workflow'], config=config.model_dump(exclude_unset=True), enabled=enabled,
                    identity=source.get('identity'))
    return db.save_server_config(scope, name, data, **options)
