"""Return existing Server.add arguments, without creating a model or worker."""


def create_entry(*, uppercase=True):
    if type(uppercase) is not bool:
        raise ValueError('uppercase must be a boolean')
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
