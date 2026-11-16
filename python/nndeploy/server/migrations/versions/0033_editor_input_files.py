"""Protect editor input resources separately from task-produced output files."""
from alembic import op
from sqlalchemy import Column, String, inspect

revision = '0033_editor_input_files'
down_revision = '0032_editor_file_fields'
branch_labels = None
depends_on = None


def upgrade():
    # Fresh databases create this table from the current metadata in 0029.
    columns = inspect(op.get_bind()).get_columns('server_editor_file_refs')
    if 'kind' not in {column['name'] for column in columns}:
        op.add_column('server_editor_file_refs', Column('kind', String(8), nullable=False, server_default='output'))


def downgrade():
    raise RuntimeError('Automatic downgrade would discard editor input file protection')
