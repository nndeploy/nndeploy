"""Preserve workflow timestamp microseconds when moving records to MySQL."""
from alembic import op
from sqlalchemy import DateTime, text
from sqlalchemy.dialects.mysql import DATETIME

revision = '0031_workflow_timestamps'
down_revision = '0030_task_inputs'
branch_labels = None
depends_on = None


def upgrade():
    if op.get_bind().dialect.name == 'mysql':
        for table in ('workflows', 'templates'):
            op.alter_column(table, 'created_at', existing_type=DateTime(), type_=DATETIME(fsp=6),
                            existing_nullable=True, server_default=text('CURRENT_TIMESTAMP(6)'))


def downgrade():
    raise RuntimeError('Automatic downgrade would lose workflow timestamp precision')
