"""Persist file ownership for invocation API authorization."""
from alembic import op
from sqlalchemy import Column, String, inspect

revision = '0007_file_owner'
down_revision = '0006_queue_control'
branch_labels = None
depends_on = None


def upgrade():
    if 'owner' not in {column['name'] for column in inspect(op.get_bind()).get_columns('server_files')}:
        op.add_column('server_files', Column('owner', String(128)))


def downgrade():
    raise RuntimeError('Automatic downgrade would remove file ownership')
