"""Persist the active shared configuration and physical runtime holders."""
from alembic import op
from nndeploy.server.shared import entries, configs, workers

revision = '0016_shared_configs'
down_revision = '0015_queue_control_tokens'
branch_labels = None
depends_on = None


def upgrade():
    for table in (entries, configs, workers):
        table.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError('Automatic downgrade would remove shared configuration ownership')
