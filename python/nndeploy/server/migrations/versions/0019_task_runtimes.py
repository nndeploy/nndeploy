"""Bind claimed tasks to a physically verifiable shared runtime."""
from alembic import op
from sqlalchemy import Column, String, inspect

revision = '0019_task_runtimes'
down_revision = '0018_control_guards'
branch_labels = None
depends_on = None


def upgrade():
    from nndeploy.server.shared import runtime_records
    runtime_records.create(op.get_bind(), checkfirst=True)
    for table, names in (('server_shared_workers', ('runtime_holder',)),
                         ('server_tasks', ('runtime_token', 'runtime_holder'))):
        existing = {column['name'] for column in inspect(op.get_bind()).get_columns(table)}
        for name in names:
            if name not in existing:
                op.add_column(table, Column(name, String(32)))
    if 'ix_server_tasks_runtime' not in {index['name'] for index in inspect(op.get_bind()).get_indexes('server_tasks')}:
        op.create_index('ix_server_tasks_runtime', 'server_tasks', ['runtime_token', 'state', 'queue_key'])


def downgrade():
    raise RuntimeError('Automatic downgrade would remove task execution ownership evidence')
