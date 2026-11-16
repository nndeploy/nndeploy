"""Archive metadata and payloads are validated without extraction."""
import io
import stat
import struct
import warnings
import zipfile

import pytest

from nndeploy.server.errors import ServerError
from nndeploy.server.storage.archive import MAX_BYTES, MAX_ENTRIES
from test_storage import local


def content(entries, compression=zipfile.ZIP_STORED):
    stream = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with zipfile.ZipFile(stream, 'w', compression=compression) as archive:
            for name, data in entries:
                archive.writestr(name, data)
    return stream.getvalue()


@pytest.mark.parametrize('compression', [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_zip_files_are_validated_stored_and_downloadable(local, compression):
    data = content([('text.txt', 'hello'), ('nested/', ''), ('nested/中文.txt', '你好')], compression)
    record = local.save(io.BytesIO(data), 'application/zip')
    path, _ = local.resolve(record['file_id'])
    assert path.suffix == '.zip' and record['media_type'] == 'application/zip'
    assert path.read_bytes() == data


@pytest.mark.parametrize('entries', [
    [('../secret', 'x')], [('/absolute', 'x')], [('C:relative', 'x')], [('folder\\file', 'x')],
    [('folder//file', 'x')], [('a/./file', 'x')], [('a', 'one'), ('a', 'two')],
    [('a', 'file'), ('a/', '')], [('a', 'file'), ('a/child', 'child')],
    [('a/child', 'child'), ('a', 'file')], [('a/' * 65, '')], [('x' * 1025, 'x')],
])
def test_zip_unsafe_or_ambiguous_names_are_not_published(local, entries):
    with pytest.raises(ServerError) as caught:
        local.save(io.BytesIO(content(entries)), 'application/zip')
    assert caught.value.code == 'invalid_archive'
    assert {path.name for path in local.root.iterdir()} == {'.server-id'}


@pytest.mark.parametrize('damage', ['truncated', 'trailing', 'crc', 'too_many', 'decoded_size', 'encrypted', 'symlink', 'fifo', 'nul_name'])
def test_zip_corruption_and_bounded_metadata_rejected(local, damage):
    data = bytearray(content([('value', 'payload')]))
    central = data.index(b'PK\x01\x02')
    end = data.index(b'PK\x05\x06')
    if damage == 'truncated':
        data = data[:-1]
    elif damage == 'trailing':
        data += b'extra'
    elif damage == 'crc':
        data[data.index(b'payload')] ^= 1
    elif damage == 'too_many':
        struct.pack_into('<HH', data, end + 8, MAX_ENTRIES + 1, MAX_ENTRIES + 1)
    elif damage == 'decoded_size':
        struct.pack_into('<I', data, central + 24, MAX_BYTES + 1)
    elif damage == 'encrypted':
        struct.pack_into('<H', data, central + 8, 1)
    elif damage == 'nul_name':
        data[central + 48] = 0
        data[32] = 0
    else:
        mode = stat.S_IFLNK if damage == 'symlink' else stat.S_IFIFO
        struct.pack_into('<I', data, central + 38, (mode | 0o600) << 16)
    with pytest.raises(ServerError) as caught:
        local.save(io.BytesIO(data), 'application/zip')
    assert caught.value.code == 'invalid_archive'
    assert {path.name for path in local.root.iterdir()} == {'.server-id'}


def test_empty_zip_is_a_valid_empty_directory(local):
    record = local.save(io.BytesIO(content([])), 'application/zip')
    with local.open(record['file_id']) as stream, zipfile.ZipFile(stream) as archive:
        assert archive.namelist() == []


def test_model_directory_extraction_validates_before_creating_any_paths(tmp_path):
    from nndeploy.server.storage.archive import extract_archive
    archive = tmp_path / 'model.zip'
    archive.write_bytes(content([('../outside', 'bad')]))
    destination = tmp_path / 'expanded'
    with pytest.raises(ValueError):
        extract_archive(archive, destination)
    assert not destination.exists() and not (tmp_path / 'outside').exists()


@pytest.mark.parametrize('entries', [
    [('A/value', 'one'), ('a/other', 'two')], [('VALUE', 'one'), ('value', 'two')],
])
def test_model_directory_extraction_never_merges_filesystem_name_aliases(tmp_path, entries):
    from nndeploy.server.storage.archive import extract_archive
    (tmp_path / 'probe').write_text('probe')
    insensitive = (tmp_path / 'PROBE').exists()
    archive = tmp_path / 'model.zip'
    archive.write_bytes(content(entries))
    destination = tmp_path / 'expanded'
    if insensitive:
        with pytest.raises((ValueError, FileExistsError)):
            extract_archive(archive, destination)
    else:
        extract_archive(archive, destination)
        for name, value in entries:
            assert (destination / name).read_text() == value


def test_configured_archive_capacity_bounds_decoded_contents_before_extraction(tmp_path):
    from nndeploy.server.storage.archive import extract_archive
    archive = tmp_path / 'model.zip'
    archive.write_bytes(content([('weights.bin', 'x' * 4096)], zipfile.ZIP_DEFLATED))
    rejected = tmp_path / 'rejected'
    with pytest.raises(ValueError, match='decoded contents'):
        extract_archive(archive, rejected, max_bytes=1024)
    assert not rejected.exists()
    accepted = tmp_path / 'accepted'
    extract_archive(archive, accepted, max_bytes=4096)
    assert (accepted / 'weights.bin').read_bytes() == b'x' * 4096
