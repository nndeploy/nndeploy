"""Idempotent conditional writes for saved workflow control operations."""
from alembic import op
from sqlalchemy import Column, String, inspect

revision = '0012_config_writes'
down_revision = '0011_file_cleanup'
branch_labels = None
depends_on = None


def upgrade():
    if 'write_token' not in {column['name'] for column in inspect(op.get_bind()).get_columns('server_configs')}:
        op.add_column('server_configs', Column('write_token', String(64)))


def downgrade():
    raise RuntimeError('Automatic downgrade would remove configuration write ownership')
