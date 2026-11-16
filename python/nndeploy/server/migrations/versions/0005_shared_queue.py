"""Durable shared queue capacity, DB identity and conditional worker ownership."""
from alembic import op
from sqlalchemy import Column, String, Double, inspect
from nndeploy.server.db import queue_limits, server_meta

revision = '0005_shared_queue'
down_revision = '0004_file_references'
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    columns = {column['name'] for column in inspect(connection).get_columns('server_tasks')}
    for column in (Column('queue_key', String(64)), Column('group_key', String(64)), Column('expires_at', Double),
                   Column('worker_token', String(64)), Column('lease_expires_at', Double), Column('published_at', Double)):
        if column.name not in columns:
            op.add_column('server_tasks', column)
    queue_limits.create(connection, checkfirst=True)
    server_meta.create(connection, checkfirst=True)
    op.create_index('ix_server_tasks_queue_state', 'server_tasks', ['queue_key', 'state', 'published_at'])


def downgrade():
    raise RuntimeError('Automatic downgrade would remove shared task ownership')
