"""Distinguish multiple file outputs from the same editor node."""
from alembic import op
from sqlalchemy import Column, Text, inspect

revision = '0032_editor_file_fields'
down_revision = '0031_workflow_timestamps'
branch_labels = None
depends_on = None


def upgrade():
    # The initial table migration uses the current metadata on fresh databases.
    columns = inspect(op.get_bind()).get_columns('server_editor_file_refs')
    if 'field_path' not in {column['name'] for column in columns}:
        op.add_column('server_editor_file_refs', Column('field_path', Text))


def downgrade():
    raise RuntimeError('Automatic downgrade would discard editor output field identities')
