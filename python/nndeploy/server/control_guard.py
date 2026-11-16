"""Physical liveness evidence for shared Server runtimes and configuration changes.

The controller and every isolated child hold a shared OS lock. Recovery needs
an exclusive lock and durably revokes it before touching database records. A
child spawned just before its parent died must acquire the lock and check the
revocation byte before it constructs any model. Missing files prove nothing.
Lock files are never unlinked: a late child must observe the same inode.
"""
from contextlib import contextmanager
from functools import lru_cache
import json
import os
from pathlib import Path
import sys


# A lost Python reference is not confirmed shutdown. Keep locks alive even if
# an application drops a Server after failed cleanup; OS exit releases them.
_held_guards = set()


@lru_cache(maxsize=1)
def _boot_id():
    """Use a kernel boot UUID, never a hostname that cloned hosts may share."""
    try:
        if sys.platform.startswith('linux'):
            return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if sys.platform == 'darwin':
            import ctypes
            value = ctypes.create_string_buffer(256)
            size = ctypes.c_size_t(len(value))
            if ctypes.CDLL(None).sysctlbyname(b'kern.bootsessionuuid', value, ctypes.byref(size), None, 0) == 0:
                return value.value.decode('ascii')
    except (OSError, UnicodeError):
        pass
    return None


def _evidence(file, token, holder):
    stat = os.fstat(file.fileno())
    return dict(token=token, holder=holder, boot=_boot_id(), device=stat.st_dev, inode=stat.st_ino)


def _lock(file, *, shared):
    file.seek(0)
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        import msvcrt
        class Overlapped(ctypes.Structure):
            _fields_ = [('internal', ctypes.c_size_t), ('internal_high', ctypes.c_size_t),
                        ('offset', wintypes.DWORD), ('offset_high', wintypes.DWORD), ('event', wintypes.HANDLE)]
        lock = ctypes.WinDLL('kernel32', use_last_error=True).LockFileEx
        lock.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                         wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped)]
        lock.restype = wintypes.BOOL
        if not lock(msvcrt.get_osfhandle(file.fileno()), 1 if shared else 3, 0, 1, 0, ctypes.byref(Overlapped())):
            code = ctypes.get_last_error()
            if code == 33:  # ERROR_LOCK_VIOLATION
                raise BlockingIOError('Control lock is still held')
            raise ctypes.WinError(code)
    else:
        import fcntl
        fcntl.flock(file, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)


class ControlGuard:
    def __init__(self, path, file):
        self.path, self.file = Path(path), file
        _held_guards.add(self)

    @classmethod
    def create(cls, directory, token, holder):
        path = Path(directory) / 'control-locks' / token
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        file = os.fdopen(fd, 'r+b', buffering=0)
        try:
            file.write(b'A' + json.dumps(_evidence(file, token, holder)).encode())
            os.fsync(file.fileno())
            if os.name != 'nt':
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            _lock(file, shared=True)
            return cls(path, file)
        except BaseException:
            file.close()
            raise

    @classmethod
    def join(cls, path):
        file = open(path, 'r+b', buffering=0)
        try:
            _lock(file, shared=True)
            file.seek(0)
            if file.read(1) != b'A':
                raise RuntimeError('Shared configuration initialization was revoked')
            return cls(path, file)
        except BaseException:
            file.close()
            raise

    def close(self):
        self.file.close()
        _held_guards.discard(self)

    @staticmethod
    @contextmanager
    def recover(directory, token, holder):
        # Only locally available, matching evidence is usable. A remote or
        # missing workspace, legacy record, or malformed marker stays held.
        if not token or len(token) != 32 or any(c not in '0123456789abcdef' for c in token):
            yield False
            return
        file = None
        try:
            file = open(Path(directory) / 'control-locks' / token, 'r+b', buffering=0)
            _lock(file, shared=False)
        except (FileNotFoundError, PermissionError, BlockingIOError):
            if file is not None:
                file.close()
            yield False
            return
        except BaseException:
            if file is not None:
                file.close()
            raise
        try:
            file.seek(0)
            state, value = file.read(1), file.read(2049)
            try:
                valid = (bool(_boot_id()) and state in (b'A', b'R') and len(value) <= 2048
                         and json.loads(value) == _evidence(file, token, holder))
            except (ValueError, UnicodeError):
                valid = False
            if valid:
                file.seek(0)
                file.write(b'R')
                os.fsync(file.fileno())
            yield valid
        finally:
            file.close()
