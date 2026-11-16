"""Adopt existing editor metadata without rewriting IDs or paths."""
from alembic import op
from nndeploy.server.db import workflows, templates

revision = '0001_workflows'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    for table in (workflows, templates):
        table.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError('Automatic downgrade would delete existing workflows')
