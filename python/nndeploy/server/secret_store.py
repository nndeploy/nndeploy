"""Encrypt request credentials before durable task admission."""
import os
import stat
import tempfile
from pathlib import Path


class SecretStore:
    def __init__(self, directory, *, key_env=None):
        from cryptography.fernet import Fernet
        if key_env:
            key = os.environ[key_env].encode()
        else:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = directory / 'request.key'
            # Publish only a fully written key, including across server processes.
            fd, temporary = tempfile.mkstemp(dir=directory, prefix='.key-')
            try:
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(Fernet.generate_key())
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    pass
            finally:
                os.unlink(temporary)
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode) or mode & 0o077:
                raise ValueError('Request secret key must be a private regular file')
            key = path.read_bytes()
        self.cipher = Fernet(key)

    def seal(self, value):
        import json
        return self.cipher.encrypt(json.dumps(value).encode()).decode()

    def open(self, value):
        import json
        return json.loads(self.cipher.decrypt(value.encode()))
