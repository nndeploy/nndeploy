"""Record object backend and content checksum alongside existing file metadata."""
from alembic import op
from sqlalchemy import inspect, Column, String

revision = '0003_file_integrity'
down_revision = '0002_tasks'
branch_labels = None
depends_on = None


def upgrade():
    columns = {column['name'] for column in inspect(op.get_bind()).get_columns('server_files')}
    for column in (Column('sha256', String(64)), Column('backend', String(128), server_default='local')):
        if column.name not in columns:
            op.add_column('server_files', column)


def downgrade():
    raise RuntimeError('Automatic downgrade would remove file integrity metadata')
