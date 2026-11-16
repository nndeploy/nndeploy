"""Storage-scoped expiry, read protection and task output ownership."""
from alembic import op
from sqlalchemy import Column, String, Double, inspect

revision = '0011_file_cleanup'
down_revision = '0010_config_files'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    columns = {column['name'] for column in inspect(bind).get_columns('server_files')}
    for column in (Column('storage_scope', String(128)), Column('purge_after', Double),
                   Column('protected_until', Double), Column('cleanup_after', Double), Column('producer_task_id', String(64))):
        if column.name not in columns:
            op.add_column('server_files', column)
    indexes = {index['name'] for index in inspect(bind).get_indexes('server_files')}
    for name, fields in [('ix_server_files_cleanup', ['storage_scope', 'state', 'purge_after']),
                         ('ix_server_files_producer', ['producer_task_id'])]:
        if name not in indexes:
            op.create_index(name, 'server_files', fields)


def downgrade():
    raise RuntimeError('Automatic downgrade would remove file lifecycle protection')
