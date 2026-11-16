"""Bind explicitly supported nodes and exposed Graph edges for each request."""

import copy
import json
import math
from contextlib import contextmanager

# Node implementation details stay here, not in workflow.server.json.
_NODE_FIELDS = {
    ("inputs", "nndeploy.basic.InputStr"): ("str_", "string"),
    ("inputs", "nndeploy.codec.PILImageDecodec"): ("path_", "image"),
    ("outputs", "nndeploy.codec.PILImageEncodec"): ("path_", "image"),
}

_NATIVE_CODECS = {
    ('inputs', 'nndeploy::codec::OpenCvImageDecode'): 'image',
    ('outputs', 'nndeploy::codec::OpenCvImageEncode'): 'image',
    ('inputs', 'nndeploy::codec::OpenCvVideoDecode'): 'video',
    ('outputs', 'nndeploy::codec::OpenCvVideoEncode'): 'video',
}


def prepare_native_workflow(workflow, config):
    """Do not open a draft's bound paths while deserializing native codecs."""
    if workflow is None:
        return None, {}
    value, originals = copy.deepcopy(workflow), {}
    for group in ('inputs', 'outputs'):
        for name, field in getattr(config, group).items():
            if field.node is None:
                continue
            node = value
            for part in field.node:
                node = next((child for child in node.get('node_repository_', []) if child['name_'] == part), {})
            if (group, node.get('key_')) in _NATIVE_CODECS:
                originals[group, name] = copy.deepcopy(node)
                node['path_'] = ''
                if group == 'outputs':
                    node['ref_path_'] = ''
    return value, originals


def prepare_native_graph(graph, config, originals):
    for group in ('inputs', 'outputs'):
        for name, field in getattr(config, group).items():
            if field.node is None:
                continue
            node = graph
            for part in field.node:
                node = node.get_node(part) if node is not None else None
            if node is None or (group, node.get_key()) not in _NATIVE_CODECS:
                continue
            if not hasattr(node, 'reset'):
                raise ValueError('Native codec bindings must be rebuilt to support resident media requests')
            original = originals.setdefault((group, name), json.loads(node.serialize()))
            _reset_codec(node, group, original)


def _check_status(status):
    import nndeploy.base
    if status != nndeploy.base.StatusCode.Ok:
        raise ValueError(f'Native media operation failed: {status}')


def _reset_codec(node, group, original):
    args = [original.get('path_', '')]
    if group == 'outputs':
        args.append(original.get('ref_path_', ''))
    _check_status(node.reset(*args))


class GraphIO:
    def __init__(self, graph, config, *, max_bytes=64 * 1024**2, native_originals=None):
        self.config = config
        self.max_bytes = max_bytes
        self.targets = {}
        self.media_kinds = {}
        self.originals = []
        self.native_codecs = {}
        self.native_originals = native_originals or {}
        self.steps = 1
        self.parameter_nodes = {}
        if not config.outputs:
            raise ValueError("Graph requires at least one declared output")
        for group in ("inputs", "parameters", "outputs"):
            for name, field in getattr(config, group).items():
                if field.node is not None:
                    node = graph
                    for part in field.node:
                        node = node.get_node(part)
                        if node is None:
                            raise ValueError(f"Unknown node: {field.node}")
                    if group == 'parameters' and field.type == 'json' and isinstance(field.default, dict):
                        self.parameter_nodes[name] = (node, json.loads(node.serialize()))
                        continue
                    mapping = _NODE_FIELDS.get((group, node.get_key()))
                    native_kind = _NATIVE_CODECS.get((group, node.get_key()))
                    if native_kind is not None:
                        if native_kind == 'image' and field.type != 'image':
                            raise ValueError('Native image codecs require image fields')
                        if native_kind == 'video' and (field.type != 'file' or field.media_type not in ('video/mp4', 'video/webm')):
                            raise ValueError('Native video codecs require file fields with video/mp4 or video/webm media_type')
                        self.native_codecs[group, name] = (node, native_kind)
                        continue
                    path_input = group in ('inputs', 'parameters') and node.get_key() == 'nndeploy.basic.InputStr' and field.type == 'file'
                    if path_input:
                        mapping = ('str_', 'file')
                    if mapping is None or mapping[1] != field.type:
                        raise ValueError(f"Unsupported {group} binding for {node.get_key()}: {name}")
                    attribute = mapping[0]
                    self.targets[group, name] = (node, attribute)
                    self.originals.append((node, attribute, copy.deepcopy(getattr(node, attribute))))
                elif field.edge is not None:
                    edges = graph.get_all_output() if group == "outputs" else graph.get_all_input()
                    if field.edge not in {edge.get_name() for edge in edges}:
                        raise ValueError(f"Not an exposed Graph {group} edge: {field.edge}")
                    self.targets[group, name] = graph.get_edge(field.edge)
                    if field.type in ('image', 'file'):
                        self.media_kinds[group, name] = self._media_kind(self.targets[group, name], field, group)
                    if field.type in ('array', 'tensor'):
                        from .arrays import check_graph_binding
                        check_graph_binding(field, self.targets[group, name])
                else:
                    raise ValueError(f"Graph field {group}.{name} requires node or edge")
        videos = [key for key, (_, kind) in self.native_codecs.items() if key[0] == 'inputs' and kind == 'video']
        if videos and (len(videos) != 1 or any(self.native_codecs.get(('outputs', name), (None, None))[1] != 'video'
                                             for name in config.outputs)):
            raise ValueError('A native video request requires one video input and video encoder outputs')
        if not videos and any(group == 'outputs' and kind == 'video' for (group, _), (_, kind) in self.native_codecs.items()):
            raise ValueError('Native video output requires a video decoder input')

    @contextmanager
    def parameters(self, graph, values, output_paths, before_init=None):
        from .parameters import merge_parameters, parameter_changes
        changes = []
        requested = {}
        for name, (node, original) in self.parameter_nodes.items():
            patch = parameter_changes(self.config.parameters[name].default, values['parameters'][name], name)
            if patch:
                changes.append((node, original, merge_parameters(original, patch)))
                requested[tuple(self.config.parameters[name].node)] = patch
        for name, field in self.config.outputs.items():
            patch = requested.get(tuple(field.node or ()), {})
            if field.is_file() and isinstance(patch.get('path_'), str):
                from pathlib import Path
                if not patch['path_']:
                    raise ValueError('Output path_ must not be empty')
                output_paths[name] = Path(patch['path_']).expanduser().resolve()
        if not changes:
            yield requested
            return

        def deserialize(node, data):
            data = copy.deepcopy(data)
            # Nested nodes are patched separately. Re-deserializing a Graph's
            # children here would also reset unrelated nodes and codec paths.
            data.pop('node_repository_', None)
            # Native codecs receive their request paths in bind(), after init;
            # deserializing an output path can otherwise truncate a draft file.
            if any(target is node for target, _ in self.native_codecs.values()):
                data['path_'] = ''
                if 'ref_path_' in data:
                    data['ref_path_'] = ''
            _check_status(node.deserialize(json.dumps(data, ensure_ascii=False)))

        _check_status(graph.deinit())
        try:
            for node, _, updated in changes:
                deserialize(node, updated)
            if before_init is not None:
                before_init(graph)
            _check_status(graph.init())
            yield requested
        finally:
            _check_status(graph.deinit())
            for node, original, _ in changes:
                deserialize(node, original)
            if before_init is not None:
                before_init(graph)
            _check_status(graph.init())

    @contextmanager
    def bind(self, graph, values, input_paths, output_paths, before_init=None):
        with self.parameters(graph, values, output_paths, before_init) as requested:
            with self._bind(graph, values, input_paths, output_paths, requested):
                yield

    @contextmanager
    def _bind(self, graph, values, input_paths, output_paths, requested):
        self.steps = 1
        video = None
        try:
            for group in ("inputs", "parameters"):
                for name, field in getattr(self.config, group).items():
                    if group == 'parameters' and name in self.parameter_nodes:
                        continue
                    if (group, name) in self.native_codecs:
                        node, kind = self.native_codecs[group, name]
                        _check_status(node.set_path(str(input_paths[group, name])))
                        if kind == 'video':
                            count, fps = node.get_size(), node.get_fps()
                            if not 1 <= count <= 18000 or not math.isfinite(fps) or not 0 < fps <= 240:
                                from .errors import ServerError
                                raise ServerError(422, 'invalid_media', 'Video requires 1–18000 frames and a finite frame rate between 0 and 240 fps')
                            self.steps, video = count, node
                        continue
                    value = values[group][name]
                    if field.type in ("image", "file"):
                        path = input_paths[group, name]
                        value = str(path) if field.node is not None else self._read_media(path, self.media_kinds[group, name])
                    elif field.type in ('array', 'tensor'):
                        from .arrays import decode
                        value = decode(field, value, path=input_paths.get((group, name)))
                    target = self.targets[group, name]
                    if field.node is not None:
                        node, attribute = target
                        setattr(node, attribute, value)
                    else:
                        import nndeploy.base
                        status = target.set(value if field.type in ('array', 'tensor', 'image', 'file') else copy.deepcopy(value))
                        if status != nndeploy.base.StatusCode.Ok:
                            raise ValueError(f'Could not set Graph input {name}: {status}')
            for name, field in self.config.outputs.items():
                if ('outputs', name) in self.native_codecs:
                    node, kind = self.native_codecs['outputs', name]
                    patch = requested.get(tuple(field.node), {})
                    if kind == 'video':
                        node.set_fps(patch.get('fps_', video.get_fps()))
                        node.set_size(patch.get('size_', self.steps))
                        if patch.get('ref_path_'):
                            _check_status(node.set_ref_path(patch['ref_path_']))
                    _check_status(node.set_path(str(output_paths[name])))
                    continue
                if field.type in ("image", "file") and field.node is not None:
                    node, attribute = self.targets["outputs", name]
                    setattr(node, attribute, str(output_paths[name]))
            yield
        finally:
            errors = []
            for (group, name), (node, _) in self.native_codecs.items():
                try:
                    _reset_codec(node, group, self.native_originals[group, name])
                except BaseException as exc:
                    errors.append(exc)
            for node, attribute, value in self.originals:
                setattr(node, attribute, copy.deepcopy(value))
            if errors:
                raise errors[0]

    def check_output_sizes(self, output_paths):
        for path in output_paths.values():
            if path.exists() and path.stat().st_size > self.max_bytes:
                raise ValueError('Graph output exceeds the file byte limit')

    def collect(self, graph, output_paths):
        result = {}
        for name, field in self.config.outputs.items():
            if field.type in ('array', 'tensor') and field.is_file():
                from .arrays import encode
                path = output_paths[name]
                with path.open('wb') as stream:
                    encode(field, self.targets['outputs', name].get_graph_output(), stream=stream, max_file_bytes=self.max_bytes)
                if path.stat().st_size > self.max_bytes:
                    raise ValueError('Graph output exceeds the file byte limit')
                result[name] = path
            elif field.type in ("image", "file"):
                path = output_paths[name]
                if ('outputs', name) in self.native_codecs:
                    node, _ = self.native_codecs['outputs', name]
                    _reset_codec(node, 'outputs', self.native_originals['outputs', name])
                if field.edge is not None:
                    value = self.targets['outputs', name].get_graph_output()
                    self._write_media(path, value, self.media_kinds['outputs', name])
                if not path.is_file():
                    raise RuntimeError(f"Output file was not produced: {name}")
                if path.stat().st_size > self.max_bytes:
                    raise ValueError('Graph output exceeds the file byte limit')
                result[name] = path
            else:
                value = self.targets["outputs", name].get_graph_output()
                if field.type in ('array', 'tensor'):
                    from .arrays import encode
                    result[name] = encode(field, value)
                else:
                    result[name] = copy.deepcopy(field.validate_value(value))
        return result

    @staticmethod
    def _media_kind(edge, field, group):
        infos = [edge.get_type_info()]
        for nodes, ports, declarations in ((edge.get_consumers(), 'get_all_input', 'get_input_type_info'),
                                           (edge.get_producers(), 'get_all_output', 'get_output_type_info')):
            for node in nodes:
                types = getattr(node, declarations)()
                for index, port in enumerate(getattr(node, ports)()):
                    if port.get_name() == edge.get_name() and index < len(types):
                        infos.append(types[index])
        kinds = set()
        for info in infos:
            if info is None:
                continue
            name, qualified = info.get_type_name(), info.get_type_name_with_namespace()
            if field.type == 'image' and qualified in ('PIL.Image', 'PIL.Image.Image'):
                kinds.add('pil')
            elif field.type == 'image' and (getattr(info, 'is_cvmat_type', lambda: False)() or qualified == 'numpy.ndarray'):
                kinds.add('opencv')
            elif field.type == 'file' and (qualified == 'builtins.bytes' or name == 'bytes'):
                kinds.add('bytes')
            elif field.type == 'file' and group != 'outputs' and (qualified == 'builtins.str' or name == 'str'):
                kinds.add('path')
            else:
                raise ValueError(f'Graph media edge type {qualified!r} does not match {field.type}')
        if len(kinds) != 1:
            raise ValueError('Media Graph ports must declare one compatible type')
        return kinds.pop()

    def _read_media(self, path, kind):
        if path.stat().st_size > self.max_bytes:
            raise ValueError('Graph input exceeds the file byte limit')
        if kind == 'path':
            return str(path)
        if kind == 'bytes':
            with path.open('rb') as source:
                data = source.read(self.max_bytes + 1)
            if len(data) > self.max_bytes:
                raise ValueError('Graph input exceeds the file byte limit')
            return data
        from .storage.media import validate
        validate(path)
        if kind == 'pil':
            from PIL import Image
            with Image.open(path) as image:
                return image.copy()
        import cv2
        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if value is None:
            raise ValueError('Could not decode Graph image input')
        return value

    def _write_media(self, path, value, kind):
        if kind == 'bytes':
            if not isinstance(value, bytes) or len(value) > self.max_bytes:
                raise ValueError('File output must contain bounded bytes')
            path.write_bytes(value)
            return
        if kind == 'pil':
            from PIL import Image
            if not isinstance(value, Image.Image) or not 0 < value.width * value.height <= 32_000_000:
                raise ValueError('Image output must contain a bounded PIL image')
            value.save(path)
            return
        import numpy as np
        import cv2
        if (not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.ndim not in (2, 3)
                or not 0 < value.shape[0] * value.shape[1] <= 32_000_000
                or value.ndim == 3 and value.shape[2] not in (3, 4)):
            raise ValueError('Image output must contain a bounded uint8 OpenCV image')
        if not cv2.imwrite(str(path), value):
            raise ValueError('Could not encode Graph image output')
