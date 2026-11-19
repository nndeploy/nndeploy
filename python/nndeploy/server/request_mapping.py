"""Explicit business fields mapped to the existing Deploy request envelope."""
from copy import deepcopy

from pydantic import BaseModel

from .protocol import DeployRequest


class RequestMapping:
    """Map validated model fields to dot-separated inputs/parameters paths.

    No implicit flattening or code expressions. The normal Server validation
    still runs after conversion, including exposed node parameter checks.
    """

    def __init__(self, model, fields):
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise TypeError('Request model must be a Pydantic BaseModel')
        if set(fields) != set(model.model_fields):
            raise ValueError('Every request model field must have exactly one mapping')
        self.model = model
        self.fields = {}
        self.input_names = {}
        for name, field in model.model_fields.items():
            alias = field.validation_alias or field.alias or name
            if not isinstance(alias, str):
                raise ValueError('Request mapping supports string field aliases; use an algorithm adapter for alias paths/choices')
            self.input_names[name] = alias
        if {'inputs', 'parameters'} & (set(model.model_fields) | set(self.input_names.values())):
            raise ValueError('Business fields must not shadow Deploy envelope names')
        if len(set(self.input_names.values())) != len(self.input_names):
            raise ValueError('Business field aliases must be distinct')
        for name, path in fields.items():
            if not isinstance(path, (str, tuple, list)):
                raise ValueError('Mapping paths must be strings or sequences of strings')
            parts = tuple(path.split('.')) if isinstance(path, str) else tuple(path)
            if len(parts) < 2 or parts[0] not in ('inputs', 'parameters') or any(not p or not isinstance(p, str) for p in parts):
                raise ValueError(f'Invalid Deploy field path: {path}')
            for previous in self.fields.values():
                if parts[:len(previous)] == previous or previous[:len(parts)] == parts:
                    raise ValueError('Request mapping targets must not overlap')
            self.fields[name] = parts

    def to_deploy(self, body, **transport):
        values = self.model.model_validate(body).model_dump(mode='json', by_alias=False)
        result = {'inputs': {}, 'parameters': {}}
        for name, parts in self.fields.items():
            target = result
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = deepcopy(values[name])
        if set(transport) & {'inputs', 'parameters'}:
            raise ValueError('Transport options cannot replace mapped fields')
        return DeployRequest.model_validate({**result, **transport})

    def from_deploy(self, body):
        """Return a business model; reject unmapped data rather than dropping it."""
        request = body if isinstance(body, DeployRequest) else DeployRequest.model_validate(body)
        remaining = deepcopy({'inputs': request.inputs, 'parameters': request.parameters})
        values = {}
        for name, parts in self.fields.items():
            target, parents = remaining, []
            for part in parts[:-1]:
                if not isinstance(target, dict) or part not in target:
                    break
                parents.append((target, part))
                target = target[part]
            else:
                if isinstance(target, dict) and parts[-1] in target:
                    values[name] = target.pop(parts[-1])
                    for parent, key in reversed(parents):
                        if parent[key] == {}:
                            del parent[key]
        if any(remaining.values()):
            raise ValueError('Deploy request contains fields without a business mapping')
        # Validate by canonical names even if the model has input aliases.
        return self.model.model_validate({self.input_names[name]: value for name, value in values.items()})

    def parse(self, body):
        """Accept either an explicit Deploy envelope or the business model."""
        if not isinstance(body, dict):
            raise ValueError('Request body must be an object')
        if {'inputs', 'parameters'} & body.keys():
            request = DeployRequest.model_validate(body)
            value = self.from_deploy(request)
            return self.to_deploy(value, **request.model_dump(exclude={'inputs', 'parameters'}))
        return self.to_deploy(body)

    def router(self, path='/request'):
        """Expose a typed business protocol through the existing get_deploy path."""
        from fastapi import APIRouter, Depends, Request
        from fastapi.exceptions import RequestValidationError
        from pydantic import ValidationError
        from .dependencies import get_deploy
        from .errors import ServerError
        router = APIRouter()

        from typing import Union

        async def submit(body: Union[self.model, DeployRequest], request: Request, deploy=Depends(get_deploy)):
            try:
                body = self.parse(await request.json())
            except ValidationError as exc:
                raise RequestValidationError(exc.errors()) from exc
            except (ValueError, TypeError) as exc:
                raise ServerError(422, 'invalid_request', str(exc)) from exc
            return await deploy(body)

        router.add_api_route(path, submit, methods=['POST'])
        return router
