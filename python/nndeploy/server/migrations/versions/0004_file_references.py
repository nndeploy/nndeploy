"""Protect task file references and prevent admission during file deletion."""
from alembic import op
import json
from sqlalchemy import inspect, Column, String, select, insert
from nndeploy.server.db import file_refs, tasks, files, _file_ids

revision = '0004_file_references'
down_revision = '0003_file_integrity'
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    columns = {column['name'] for column in inspect(connection).get_columns('server_files')}
    if 'state' not in columns:
        op.add_column('server_files', Column('state', String(16), nullable=False, server_default='ready'))
    file_refs.create(connection, checkfirst=True)
    for row in connection.execute(select(tasks.c.task_id, tasks.c.inputs, tasks.c.parameters, tasks.c.outputs)):
        values = [json.loads(value) for value in (row.inputs, row.parameters, row.outputs) if value is not None]
        for identity in _file_ids(values):
            if connection.execute(select(files.c.file_id).where(files.c.file_id == identity)).first():
                connection.execute(insert(file_refs).values(file_id=identity, task_id=row.task_id))


def downgrade():
    raise RuntimeError('Automatic downgrade would remove file reference protection')
