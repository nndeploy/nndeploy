"""Per-task retention with one atomic expiry of results, events and references."""
from alembic import op
from sqlalchemy import Column, Double, inspect

revision = '0009_task_retention'
down_revision = '0008_task_events'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    columns = {column['name'] for column in inspect(bind).get_columns('server_tasks')}
    for name in ('retention_seconds', 'purge_after'):
        if name not in columns:
            op.add_column('server_tasks', Column(name, Double))
    if 'ix_server_tasks_purge' not in {index['name'] for index in inspect(bind).get_indexes('server_tasks')}:
        op.create_index('ix_server_tasks_purge', 'server_tasks', ['purge_after', 'task_id'])


def downgrade():
    raise RuntimeError('Automatic downgrade would remove task retention settings')
