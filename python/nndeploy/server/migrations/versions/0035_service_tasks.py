"""Algorithm protocol metadata and atomic external identity/admission."""
from alembic import op
from sqlalchemy import Column, String, Text, Integer, Double, inspect
revision = '0035_service_tasks'
down_revision = '0034_task_callbacks'
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    columns = {c['name'] for c in inspect(connection).get_columns('server_tasks')}
    for column in (Column('service', Text), Column('service_scope', String(64)), Column('external_key', String(64)), Column('admission_key', String(64)),
                   Column('progress_sequence', Integer, nullable=False, server_default='0'),
                   Column('progress_next_at', Double, nullable=False, server_default='0'),
                   Column('progress_delivery', Text)):
        if column.name not in columns:
            op.add_column('server_tasks', column)
    indexes = {i['name'] for i in inspect(connection).get_indexes('server_tasks')}
    if 'uq_server_tasks_external' not in indexes:
        op.create_index('uq_server_tasks_external', 'server_tasks', ['external_key'], unique=True)
    if 'ix_server_tasks_admission' not in indexes:
        op.create_index('ix_server_tasks_admission', 'server_tasks', ['admission_key', 'state'])


def downgrade():
    raise RuntimeError('Downgrade would discard external task identities')
