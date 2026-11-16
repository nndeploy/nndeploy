"""Protect managed output files while their editor task history is retained."""
from alembic import op

revision = '0029_editor_output_files'
down_revision = '0028_http_audit'
branch_labels = None
depends_on = None


def upgrade():
    from nndeploy.server.db import editor_file_refs
    editor_file_refs.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError('Automatic downgrade would discard editor output references')
