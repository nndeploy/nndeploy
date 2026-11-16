"""Persist editor admission and physical execution ownership."""
from alembic import op
from sqlalchemy import BigInteger, Boolean, Column, Double, Index, Integer, MetaData, String, Table, Text
from sqlalchemy.dialects.mysql import LONGTEXT

revision = '0025_editor_tasks'
down_revision = '0024_editor_history'
branch_labels = None
depends_on = None


def upgrade():
    Table('server_editor_tasks', MetaData(),
        Column('scope', String(64), primary_key=True), Column('task_id', String(64), primary_key=True),
        Column('token', String(32), nullable=False), Column('holder', String(32), nullable=False),
        Column('state', String(16), nullable=False), Column('created_at', Double, nullable=False),
        Column('sequence', BigInteger, nullable=False), Column('priority', Integer, nullable=False),
        Column('payload', Text().with_variant(LONGTEXT(), 'mysql'), nullable=False),
        Column('cancel_requested', Boolean, nullable=False, server_default='0'),
        Column('dispatched_at', Double), Column('started_at', Double), Column('worker_pid', BigInteger),
        Index('ix_server_editor_tasks_recovery', 'scope', 'state', 'sequence')).create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError('Automatic downgrade would remove editor execution ownership')
