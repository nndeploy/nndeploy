"""Request overrides use the node's existing serialized parameter structure."""
import copy


_STRUCTURE = {
    'key_', 'name_', 'inputs_', 'outputs_', 'node_repository_', 'node_type_', 'io_type_',
    'desc_', 'developer_', 'source_', 'version_', 'required_params_', 'dropdown_params_',
    'io_params_', 'io_param_', 'ui_params_', 'nndeploy_ui_layout', 'size', 'is_graph_', 'is_inner_',
    'is_dynamic_input_', 'is_dynamic_output_', 'control_dependencies', 'execution_order',
    'parallel_type_', 'loop_count_', 'is_loop_max_flag_', 'is_external_stream_',
    'is_graph_node_share_stream_',
}


def node_parameters(node):
    return {key: copy.deepcopy(value) for key, value in node.items() if key not in _STRUCTURE}


def parameter_changes(defaults, supplied, path='parameters'):
    if not isinstance(supplied, dict):
        raise ValueError(f'{path} must be an object')
    changes = {}
    for key, value in supplied.items():
        if key not in defaults:
            raise ValueError(f'Unknown node parameter: {path}.{key}')
        original = defaults[key]
        location = f'{path}.{key}'
        if isinstance(original, dict):
            changed = parameter_changes(original, value, location)
            if changed:
                changes[key] = changed
            continue
        valid = original is None or type(value) is type(original) or (
            type(original) is float and type(value) is int)
        if not valid:
            raise ValueError(f'{location} must have type {type(original).__name__}')
        if value != original:
            changes[key] = copy.deepcopy(float(value) if type(original) is float else value)
    return changes


def merge_parameters(target, changes):
    result = copy.deepcopy(target)
    for key, value in changes.items():
        result[key] = (merge_parameters(result[key], value)
                       if isinstance(value, dict) and isinstance(result.get(key), dict)
                       else copy.deepcopy(value))
    return result


def parameter_paths(value, prefix=()):
    for key, item in value.items():
        path = (*prefix, key)
        if isinstance(item, dict) and item:
            yield from parameter_paths(item, path)
        else:
            yield list(path)


def exposed_defaults(defaults, exposed):
    if exposed is None:
        return copy.deepcopy(defaults)
    result = {}
    for path in exposed:
        source, target = defaults, result
        for key in path[:-1]:
            source = source[key]
            target = target.setdefault(key, {})
        target[path[-1]] = copy.deepcopy(source[path[-1]])
    return result


def split_node_request(config, body):
    """Translate the public node-shaped body to existing execution bindings."""
    inputs, parameters = {}, {}
    for name, value in body.items():
        field = config.parameters.get(name)
        if field is not None and isinstance(value, dict) and not (
                name in config.inputs and set(value) <= {'path', 'file_id'}):
            parameters[name] = value
        else:
            inputs[name] = value
    return inputs, parameters
