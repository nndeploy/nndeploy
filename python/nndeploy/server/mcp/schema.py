"""Translate current entry declarations without exposing worker bindings."""
import copy


def object_schema(properties, required=()):
    return dict(type='object', properties=properties, required=list(required), additionalProperties=False)


def field_schema(field):
    kind = field['type']
    if kind in ('image', 'file') or field.get('media_type') in (
            'application/x-npy', 'audio/wav', 'audio/mpeg', 'audio/ogg', 'audio/flac', 'audio/mp4'):
        result = object_schema({'file_id': {'type': 'string', 'minLength': 1}}, ['file_id'])
    elif kind in ('array', 'tensor'):
        props = {'dtype': {'const': field['dtype']},
                 'shape': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0}},
                 'data': {'type': 'array', 'items': {'type': 'boolean' if field['dtype'] == 'bool' else 'number'}}}
        if field.get('data_format'):
            props['data_format'] = {'type': 'string'}
        result = object_schema(props, ['dtype', 'shape', 'data'])
    elif kind == 'json':
        result = {}
    else:
        result = {'type': kind}
        for key in ('minimum', 'maximum'):
            if field.get(key) is not None:
                result[key] = field[key]
    if 'default' in field:
        result['default'] = copy.deepcopy(field['default'])
    return result


def tool_schema(entry):
    model = entry['source'].get('input_model')
    groups = {}
    for group in ('inputs', 'parameters'):
        fields = entry['schema'][group]
        groups[group] = object_schema({name: field_schema(field) for name, field in fields.items()},
                                      [name for name, field in fields.items() if field.get('required', True)])
    if model is not None:
        inputs = copy.deepcopy(model.model_json_schema())
        # Pydantic refs are rooted at the complete MCP arguments schema.
        definitions = inputs.pop('$defs', None)
        groups['inputs'] = inputs
    else:
        definitions = None
    required = [group for group in groups if groups[group].get('required')]
    groups['idempotency_key'] = {'type': 'string', 'minLength': 1, 'maxLength': 128}
    result = object_schema(groups, required)
    if definitions:
        result['$defs'] = definitions
    return result
