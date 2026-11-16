"""Ordered duplex input belongs to existing Tasks and follows their retention."""
from alembic import op
from sqlalchemy import Column, Text, inspect

revision = '0030_task_inputs'
down_revision = '0029_editor_output_files'
branch_labels = None
depends_on = None


def upgrade():
    from nndeploy.server.db import task_inputs
    bind = op.get_bind()
    if 'input_state' not in {column['name'] for column in inspect(bind).get_columns('server_tasks')}:
        op.add_column('server_tasks', Column('input_state', Text))
    task_inputs.create(bind, checkfirst=True)


def downgrade():
    raise RuntimeError('Automatic downgrade would discard task input streams')
