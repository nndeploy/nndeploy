"""Read canonical runtime environment variables without legacy fallbacks."""
import os


def get_env(name, default=None, *, environ=None):
    source = os.environ if environ is None else environ
    return source.get(name, default)
