"""Keep the original HTTP correlation ID with local and shared tasks."""
from alembic import op
from sqlalchemy import Column, String, inspect

revision = '0014_request_ids'
down_revision = '0013_task_payloads'
branch_labels = None
depends_on = None


def upgrade():
    if 'request_id' not in {column['name'] for column in inspect(op.get_bind()).get_columns('server_tasks')}:
        op.add_column('server_tasks', Column('request_id', String(36)))


def downgrade():
    raise RuntimeError('Automatic downgrade would remove task request correlation')
