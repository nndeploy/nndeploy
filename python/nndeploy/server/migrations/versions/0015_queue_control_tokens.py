"""Fence delayed shared control writes without changing task identities."""
from alembic import op
from sqlalchemy import Column, String, inspect

revision = '0015_queue_control_tokens'
down_revision = '0014_request_ids'
branch_labels = None
depends_on = None


def upgrade():
    if 'control_token' not in {column['name'] for column in inspect(op.get_bind()).get_columns('server_queue_limits')}:
        op.add_column('server_queue_limits', Column('control_token', String(32), nullable=False, server_default='0'))


def downgrade():
    raise RuntimeError('Automatic downgrade would remove shared control fencing')
