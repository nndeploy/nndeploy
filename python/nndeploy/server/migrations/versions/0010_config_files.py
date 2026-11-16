"""Atomic saved workflows and runtime resource references."""
from alembic import op

revision = '0010_config_files'
down_revision = '0009_task_retention'
branch_labels = None
depends_on = None


def upgrade():
    from nndeploy.server.db import server_configs, config_file_refs, file_holders, file_holds
    for table in (server_configs, config_file_refs, file_holders, file_holds):
        table.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError('Automatic downgrade would remove saved workflow resource protection')
