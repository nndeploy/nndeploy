"""Persistent Deploy requests and files, including early local schema adoption."""
from alembic import op
from sqlalchemy import inspect, Column, Text, String, Integer, Boolean
from nndeploy.server.db import tasks, files

revision = '0002_tasks'
down_revision = '0001_workflows'
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    tasks.create(connection, checkfirst=True)
    files.create(connection, checkfirst=True)
    columns = {column['name'] for column in inspect(connection).get_columns('server_tasks')}
    missing = [Column('inputs', Text), Column('parameters', Text), Column('owner', String(128)),
               Column('idempotency_key', String(128)), Column('fingerprint', String(64)),
               Column('attempt', Integer, nullable=False, server_default='1'),
               Column('cancel_requested', Boolean, nullable=False, server_default='0')]
    for column in missing:
        if column.name not in columns:
            op.add_column('server_tasks', column)
    if 'idempotency_key' not in columns:
        op.create_index('uq_server_tasks_idempotency', 'server_tasks', ['name', 'idempotency_key'], unique=True)


def downgrade():
    raise RuntimeError('Automatic downgrade would delete task records')
