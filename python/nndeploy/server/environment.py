"""Load optional local Server environment values without executing shell code."""
import os
from pathlib import Path


def load_server_env(path='server.env'):
    """Read the exact working-directory file; existing environment values win.

    Interpolation is disabled so secrets containing dollar signs or ${...}
    remain literal. Do not search parent directories or load generic .env files.
    """
    try:
        stream = Path(path).open(encoding='utf-8-sig')
    except FileNotFoundError:
        return False
    with stream:
        from dotenv import dotenv_values
        values = dotenv_values(stream=stream, interpolate=False)
    for name, value in values.items():
        if value is not None:
            os.environ.setdefault(name, value)
    return True
