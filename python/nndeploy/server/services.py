"""Load named algorithm instances through the application-local Registry."""
from pydantic import Field, model_validator

from .config import StrictModel, load_json
from .registry import ServiceAdapter


# Compatibility registrations are lazy. Algorithm options and fixed-route
# restrictions belong to these factories, not to the manifest format.
BUILTIN_SERVICES = {
    'vsr': 'nndeploy.e_dit.algorithms.vsr.server.manifest:create_service',
    'videoedit': 'nndeploy.e_dit.algorithms.wan.server.manifest:create_service',
}


class ServiceDefinition(StrictModel):
    algorithm: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')
    name: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')
    options: dict = Field(default_factory=dict)


class ServicesConfig(StrictModel):
    services: list[ServiceDefinition] = Field(min_length=1, max_length=128)

    @model_validator(mode='after')
    def validate_entries(self):
        if len({entry.name for entry in self.services}) != len(self.services):
            raise ValueError('Service names must be unique')
        return self


def load_services(path):
    return ServicesConfig.model_validate(load_json(path))


def register_services(server, config):
    """Prepare all factories before registering entries on the same runtime.

    Factories must not start models or workers. The launcher closes the runtime
    if installation fails (for example, on a conflicting custom route).
    """
    config = ServicesConfig.model_validate(config)
    if server.started or server.starting_task is not None or server.shutting_down:
        raise ValueError('Register services before starting the Server')
    if any(entry.name in server.entries for entry in config.services):
        raise ValueError('Service name is already registered')
    for name, factory in BUILTIN_SERVICES.items():
        if name not in server._registry.factories:
            server.register(name, factory)
    # Resolve every name before executing any trusted factory.
    for entry in config.services:
        if entry.algorithm not in server._registry.factories:
            raise ValueError(f'Unknown registered algorithm: {entry.algorithm}')
    prepared = []
    routes = set()
    exclusive = set(getattr(server, '_service_adapter_keys', ()))
    for entry in config.services:
        definition = server._registry.create(entry.algorithm, entry.options)
        if isinstance(definition, ServiceAdapter) and definition.exclusive_key is not None:
            if definition.exclusive_key in exclusive:
                raise ValueError(f'Service adapter already registered: {definition.exclusive_key}')
            exclusive.add(definition.exclusive_key)
        if not isinstance(definition, ServiceAdapter):
            definition = server._prepare_entry(name=entry.name, **definition)
            mounted = definition[1]
            if mounted is not None:
                conflicts = routes & mounted.nndeploy_route_keys
                if conflicts:
                    raise ValueError(f'Algorithm route conflicts with another service: {sorted(conflicts)}')
                routes.update(mounted.nndeploy_route_keys)
        prepared.append((entry.name, definition))
    for name, definition in prepared:
        if isinstance(definition, ServiceAdapter):
            server._add_registered(name, definition)
        else:
            server._commit_entry(*definition)
    return server
