"""Return existing Server.add arguments, without creating a model or worker."""


def create_entry(*, uppercase=True):
    from .api import router
    from .executor import deploy
    return {
        'deploy': deploy,
        'router': router,
        'config': {
            'inputs': {'text': {'type': 'string'}},
            'parameters': {'uppercase': {'type': 'boolean', 'required': False, 'default': uppercase}},
            'outputs': {'text': {'type': 'string'}},
        },
    }
