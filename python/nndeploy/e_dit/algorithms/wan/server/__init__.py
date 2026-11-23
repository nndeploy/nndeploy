"""Wan-owned API over the common Server task and Worker implementation."""
from .schemas import VideoRepairRequest, parse_request
from .policy import resolve_variant, apply_variant_policy


def add_videoedit(server, models, *, name='videoedit', contract='gateway', storage=None,
                  secret_key_env=None, execution=None, graph_factory=None, audit_directory=None,
                  memory_estimator=None):
    """Register normal/DMD model configs and the source HTTP contract.

    models maps normal/dmd to DiTParams or plain DiTParams dictionaries.
    Both variants share one resident Worker and one durable admission group.
    """
    from .api import VideoEditAPI
    from .service import HandlerFactory
    from nndeploy.server.secret_store import SecretStore
    if not models or set(models) - {'normal', 'dmd'}:
        raise ValueError('VideoEdit models must contain normal and/or dmd configurations')
    configured = {}
    for variant, value in models.items():
        options = value.to_dict() if hasattr(value, 'to_dict') else dict(value)
        expected = 'wan_videoedit_dmd' if variant == 'dmd' else 'wan_videoedit'
        if options.get('model', expected) != expected:
            raise ValueError(f'{variant} requires {expected}')
        options['model'] = expected
        configured[variant] = options
    # Construct the private key before workers start, so no creation race exists.
    secrets_directory = str(server.data_dir / 'request-secrets')
    SecretStore(secrets_directory, key_env=secret_key_env)
    api = VideoEditAPI(server, name, configured, contract=contract, storage=storage, key_env=secret_key_env,
                       memory_estimator=memory_estimator)
    from nndeploy.server.api.videos import prepare_video_resources
    resources = prepare_video_resources(server)
    router = api.router()
    permissions = {(method, route.path): ('tasks:cancel' if method == 'DELETE' else
                    'files:read' if route.path.endswith('/content') else 'tasks:read')
                   for route in router.routes for method in route.methods if method != 'POST'}
    settings = dict(isolation='process', workers=1, max_queue_size=0, startup_timeout_seconds=1800,
                    admission_group=api.group)
    settings.update(execution or {})
    settings.setdefault('cooperative_cancel', settings['isolation'] == 'process')
    import sys
    if settings['isolation'] == 'process' and sys.platform.startswith('linux'):
        settings.setdefault('allow_child_processes', True)
    if settings.get('workers') != 1 or settings.get('max_queue_size') != 0 or settings.get('scheduling', 'serial') != 'serial':
        raise ValueError('VideoEdit requires one serial worker with no waiting queue')
    settings['admission_group'] = api.group
    factory = HandlerFactory(configured, secrets_directory, secret_key_env,
                             graph_factory=graph_factory, audit_directory=audit_directory)
    server.add(name=name, handler_factory=factory, config={
        'inputs': {'request': {'type': 'json'}, 'storage_secret': {'type': 'json', 'required': False, 'default': None}},
        'execution': settings,
        'streaming': {'enabled': True, 'max_duration_seconds': 86400, 'max_events': 100000}},
        router=router, router_prefix='', router_permissions=permissions)
    resources.register(api)
    return server


__all__ = ['VideoRepairRequest', 'parse_request', 'resolve_variant', 'apply_variant_policy', 'add_videoedit']
