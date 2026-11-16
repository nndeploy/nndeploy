"""Persist callback delivery alongside task admission and terminal results."""
from alembic import op
from sqlalchemy import Column, String, Text, Integer, Double, inspect

revision = '0034_task_callbacks'
down_revision = '0033_editor_input_files'
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    columns = {column['name'] for column in inspect(connection).get_columns('server_tasks')}
    for column in [Column('callback_url', Text), Column('callback_scope', String(64)),
                   Column('callback_state', String(16)),
                   Column('callback_attempts', Integer, nullable=False, server_default='0'),
                   Column('callback_next_at', Double, nullable=False, server_default='0'),
                   Column('callback_status_code', Integer), Column('callback_error', Text)]:
        if column.name not in columns:
            op.add_column('server_tasks', column)
    if 'ix_server_tasks_callbacks' not in {item['name'] for item in inspect(connection).get_indexes('server_tasks')}:
        op.create_index('ix_server_tasks_callbacks', 'server_tasks', ['callback_scope', 'callback_state', 'callback_next_at'])


def downgrade():
    raise RuntimeError('Automatic downgrade would discard callback delivery state')
