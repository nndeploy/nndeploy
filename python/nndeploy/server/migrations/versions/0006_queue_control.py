"""Coordinate shared entry admission when starting and stopping workers."""
from alembic import op
from sqlalchemy import inspect, Column, Boolean

revision = '0006_queue_control'
down_revision = '0005_shared_queue'
branch_labels = None
depends_on = None


def upgrade():
    if 'accepting' not in {column['name'] for column in inspect(op.get_bind()).get_columns('server_queue_limits')}:
        op.add_column('server_queue_limits', Column('accepting', Boolean, nullable=False, server_default='1'))


def downgrade():
    raise RuntimeError('Automatic downgrade would remove shared entry controls')
