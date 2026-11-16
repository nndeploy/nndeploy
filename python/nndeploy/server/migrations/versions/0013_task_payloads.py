"""Keep bounded numeric requests/results beyond MySQL TEXT's 64 KiB limit."""
from alembic import op
from sqlalchemy import inspect
from sqlalchemy.dialects.mysql import LONGTEXT

revision = '0013_task_payloads'
down_revision = '0012_config_writes'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == 'mysql':
        for column in inspect(bind).get_columns('server_tasks'):
            if column['name'] in ('inputs', 'parameters', 'outputs', 'error') and not isinstance(column['type'], LONGTEXT):
                op.alter_column('server_tasks', column['name'], type_=LONGTEXT(), existing_type=column['type'],
                                existing_nullable=column['nullable'])


def downgrade():
    raise RuntimeError('Automatic downgrade could truncate task payloads')
