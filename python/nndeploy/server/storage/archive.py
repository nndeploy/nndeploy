"""Bounded directory archives, including explicit model-directory extraction."""
from contextlib import contextmanager
import os
import stat
import struct
import tempfile
import zipfile

MAX_ENTRIES = 4096
MAX_BYTES = 64 * 1024**2
MAX_NAME_BYTES = 1024
MAX_DEPTH = 64


def _name(value):
    name = value[:-1] if value.endswith('/') else value
    parts = name.split('/')
    if (not name or len(value.encode('utf-8')) > MAX_NAME_BYTES or len(parts) > MAX_DEPTH or
            any(part in ('', '.', '..') for part in parts) or any(char in value for char in ('\\', ':', '\x00'))):
        raise ValueError('Archive paths must be bounded relative names')
    return value


class _BoundedOutput:
    def __init__(self, stream, limit):
        self.stream, self.limit = stream, limit

    def write(self, value):
        if self.stream.tell() + len(value) > self.limit:
            raise ValueError('Directory archive exceeds the file byte limit')
        return self.stream.write(value)

    def seek(self, offset, whence=0):
        position = self.stream.seek(offset, whence)
        if not 0 <= position <= self.limit:
            raise ValueError('Directory archive exceeds the file byte limit')
        return position

    def tell(self):
        return self.stream.tell()

    def flush(self):
        self.stream.flush()


def _version(info):
    return info.st_size, info.st_mtime_ns, info.st_ctime_ns


@contextmanager
def directory_archive(descriptor, *, max_bytes, temp_dir):
    """Archive a held directory without following links below its root."""
    if os.open not in os.supports_dir_fd or os.scandir not in os.supports_fd or not hasattr(os, 'O_NOFOLLOW'):
        raise ValueError('Directory capture requires descriptor-relative filesystem operations')
    limit = min(max_bytes, MAX_BYTES)
    remaining, count = limit, 0

    def visit(directory, prefix, archive):
        nonlocal remaining, count
        before = os.fstat(directory)
        names = []
        with os.scandir(directory) as entries:
            for entry in entries:
                count += 1
                if count > MAX_ENTRIES:
                    raise ValueError('Directory has too many archive entries')
                names.append(entry.name)
        for name in sorted(names):
            relative = _name(prefix + name)
            child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, 'O_NONBLOCK', 0), dir_fd=directory)
            try:
                info = os.fstat(child)
                if stat.S_ISDIR(info.st_mode):
                    relative = _name(relative + '/')
                    archive.writestr(relative, b'')
                    visit(child, relative, archive)
                elif stat.S_ISREG(info.st_mode):
                    if info.st_size > remaining:
                        raise ValueError('Directory contents exceed the file byte limit')
                    size = 0
                    with archive.open(relative, 'w') as output:
                        while chunk := os.read(child, min(65536, remaining + 1)):
                            remaining -= len(chunk)
                            if remaining < 0:
                                raise ValueError('Directory contents exceed the file byte limit')
                            size += len(chunk)
                            output.write(chunk)
                    if size != info.st_size or _version(info) != _version(os.fstat(child)):
                        raise ValueError('Directory output changed while being captured')
                else:
                    raise ValueError('Directory outputs contain only regular files and directories')
            finally:
                os.close(child)
        if _version(before) != _version(os.fstat(directory)):
            raise ValueError('Directory output changed while being captured')

    # Anonymous on supported platforms: process termination also closes and
    # removes the temporary archive. Storage owns the final quota/reservation.
    with tempfile.TemporaryFile(dir=temp_dir) as stream:
        with zipfile.ZipFile(_BoundedOutput(stream, limit), 'w', compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
            visit(descriptor, '', archive)
        stream.seek(0)
        yield stream


def validate_archive(path, *, max_bytes=None):
    """Check metadata bounds before ZipFile allocates its central directory."""
    limit = MAX_BYTES if max_bytes is None else max_bytes
    if type(limit) is not int or not 1 <= limit <= 5 * 1024**3:
        raise ValueError('ZIP capacity must be a positive bounded byte count')
    size = path.stat().st_size
    if not 22 <= size <= limit:
        raise ValueError('ZIP size exceeds supported limits')
    with path.open('rb') as stream:
        stream.seek(max(0, size - 65557))
        tail = stream.read(65557)
    offset = tail.rfind(b'PK\x05\x06')
    if offset < 0 or len(tail) - offset < 22:
        raise ValueError('ZIP end record is missing')
    _, disk, start_disk, disk_count, count, directory_size, directory_offset, comment_size = struct.unpack(
        '<4s4H2IH', tail[offset:offset + 22])
    end = size - len(tail) + offset
    if (disk or start_disk or disk_count != count or count > MAX_ENTRIES or
            directory_offset + directory_size != end or offset + 22 + comment_size != len(tail) or
            directory_size > count * (46 + MAX_NAME_BYTES + 1024)):
        raise ValueError('ZIP requires a complete bounded single-volume directory')
    total, names, file_names, directories = 0, set(), set(), set()
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) != count:
            raise ValueError('ZIP entry count does not match its directory')
        for entry in entries:
            name = _name(entry.filename)
            if (entry.orig_filename != name or name.rstrip('/') in names or entry.flag_bits & 1 or
                    entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                raise ValueError('ZIP contains duplicate, encrypted or unsupported entries')
            names.add(name.rstrip('/'))
            mode = stat.S_IFMT(entry.external_attr >> 16)
            if (mode not in (0, stat.S_IFREG, stat.S_IFDIR) or entry.is_dir() and entry.file_size or
                    mode == stat.S_IFDIR and not entry.is_dir() or mode == stat.S_IFREG and entry.is_dir()):
                raise ValueError('ZIP entries must be regular files or empty directories')
            parts = name.rstrip('/').split('/')
            parents = {'/'.join(parts[:index]) for index in range(1, len(parts))}
            if parents & file_names or not entry.is_dir() and name in directories:
                raise ValueError('ZIP file and directory paths conflict')
            directories.update(parents)
            if entry.is_dir():
                directories.add(name.rstrip('/'))
            else:
                file_names.add(name)
            total += entry.file_size
            if total > limit:
                raise ValueError('ZIP decoded contents exceed the byte limit')
            read = 0
            with archive.open(entry) as content:
                while chunk := content.read(min(65536, entry.file_size - read + 1)):
                    read += len(chunk)
                    if read > entry.file_size:
                        raise ValueError('ZIP entry exceeds its declared size')
            if read != entry.file_size:
                raise ValueError('ZIP entry is incomplete')


def extract_archive(path, destination, *, max_bytes=None):
    """Extract a validated model directory into a new, privately owned path."""
    validate_archive(path, max_bytes=max_bytes)
    destination.mkdir()
    directories = {}

    def directory(parts):
        current = destination
        for index, part in enumerate(parts):
            current = current / part
            current.mkdir(exist_ok=True)
            info = current.stat()
            identity = info.st_dev, info.st_ino
            spelling = tuple(parts[:index + 1])
            if identity in directories and directories[identity] != spelling:
                raise ValueError('ZIP directory names alias on this filesystem')
            directories[identity] = spelling
        return current

    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            parts = _name(entry.filename).rstrip('/').split('/')
            if entry.is_dir():
                directory(parts)
            else:
                target = directory(parts[:-1]) / parts[-1]
                # Exclusive creation also rejects case/Unicode aliases on
                # filesystems which consider distinct ZIP names equivalent.
                with target.open('xb') as output, archive.open(entry) as source:
                    while chunk := source.read(65536):
                        output.write(chunk)
