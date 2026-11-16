"""Persist existing editor history without changing workflow or Deploy records."""
from alembic import op
from sqlalchemy import Column, Double, Index, MetaData, String, Table, Text
from sqlalchemy.dialects.mysql import LONGTEXT

revision = '0024_editor_history'
down_revision = '0023_temporary_memory'
branch_labels = None
depends_on = None


def upgrade():
    table = Table('server_editor_history', MetaData(),
        Column('scope', String(64), primary_key=True),
        Column('task_id', String(64), primary_key=True),
        Column('finished_at', Double, nullable=False),
        Column('record', Text().with_variant(LONGTEXT(), 'mysql'), nullable=False),
        Index('ix_server_editor_history_finished', 'scope', 'finished_at', 'task_id'))
    table.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError('Automatic downgrade would remove editor task history')
