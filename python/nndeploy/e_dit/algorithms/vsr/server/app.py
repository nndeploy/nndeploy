"""Register VSR on a Server owned by the application."""
from functools import partial
import sys

from .handler import VSRHandler
from .router import create_router
from .service import VSRService


def add_vsr(server, params, *, name='vsr', isolation='process', handler_factory=None,
            secret_key_env=None, execution=None, memory_estimator=None):
    from nndeploy.e_dit import DiTParams
    from nndeploy.server.api.videos import prepare_video_resources
    params = params if isinstance(params, DiTParams) else DiTParams(**params)
    if params.model != 'wan_vsr':
        raise ValueError('VSR endpoints require model="wan_vsr"')
    service = VSRService(server, params.build_config(), name=name, key_env=secret_key_env,
                         memory_estimator=memory_estimator)
    resources = prepare_video_resources(server)
    settings = dict(isolation=isolation, workers=1, max_queue_size=0,
                    startup_timeout_seconds=1800, shutdown_timeout_seconds=180)
    settings.update(execution or {})
    settings.setdefault('cooperative_cancel', settings['isolation'] == 'process')
    if settings['isolation'] == 'process' and sys.platform.startswith('linux'):
        settings.setdefault('allow_child_processes', True)
    if settings.get('workers') != 1 or settings.get('max_queue_size') != 0 or settings.get('scheduling', 'serial') != 'serial':
        raise ValueError('VSR requires one serial worker with no waiting queue')
    settings['admission_group'] = service.group
    server.add(name=name, handler_factory=partial(handler_factory or VSRHandler, params.to_dict(),
        str(server.data_dir / 'request-secrets'), key_env=secret_key_env),
        config={'inputs': {'request': {'type': 'json'},
                          'storage_secret': {'type': 'json', 'required': False, 'default': None}},
                'execution': settings,
                'streaming': {'enabled': True, 'max_duration_seconds': 86400, 'max_events': 100000}},
        router=create_router(service), router_prefix='')
    resources.register(service)
    server.app.state.vsr = service
    return server
