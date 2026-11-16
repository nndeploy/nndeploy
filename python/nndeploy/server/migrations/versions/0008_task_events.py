"""Durable ordered task output and terminal events."""
from alembic import op
from sqlalchemy import Column, Integer, inspect

revision = '0008_task_events'
down_revision = '0007_file_owner'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if 'event_seq' not in {column['name'] for column in inspect(bind).get_columns('server_tasks')}:
        op.add_column('server_tasks', Column('event_seq', Integer, nullable=False, server_default='0'))
    from nndeploy.server.db import task_events
    task_events.create(bind, checkfirst=True)


def downgrade():
    raise RuntimeError('Automatic downgrade would remove task events')
