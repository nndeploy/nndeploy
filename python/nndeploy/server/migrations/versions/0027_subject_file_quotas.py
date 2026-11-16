"""Index committed files and unfinished uploads by exact identity."""
import hashlib
import json

from alembic import op
from sqlalchemy import Column, MetaData, String, Table, bindparam, inspect, select, update

revision = '0027_subject_file_quotas'
down_revision = '0026_subject_task_quotas'
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    if 'owner_key' not in {c['name'] for c in inspect(connection).get_columns('server_files')}:
        op.add_column('server_files', Column('owner_key', String(64)))
    files = Table('server_files', MetaData(), Column('file_id', String(64)),
                  Column('owner', String(128)), Column('owner_key', String(64)))
    while True:
        rows = connection.execute(select(files.c.file_id, files.c.owner).where(
            files.c.owner_key.is_(None)).order_by(files.c.file_id).limit(1000)).all()
        if not rows:
            break
        connection.execute(update(files).where(files.c.file_id == bindparam('_file')).values(owner_key=bindparam('_key')),
            [{'_file': row.file_id, '_key': hashlib.sha256(json.dumps(row.owner, ensure_ascii=False).encode()).hexdigest()}
             for row in rows])
    if 'ix_server_files_owner' not in {i['name'] for i in inspect(connection).get_indexes('server_files')}:
        op.create_index('ix_server_files_owner', 'server_files', ['owner_key'])


def downgrade():
    raise RuntimeError('Automatic downgrade would remove identity file quota accounting')
