"""Algorithm code, called on the existing Server worker."""


def deploy(inputs, *, uppercase=True):
    text = inputs['text']
    return {'text': text.upper() if uppercase else text.lower()}
