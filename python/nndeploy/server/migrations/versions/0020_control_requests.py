"""Durable idempotency receipts for existing management HTTP operations."""
from alembic import op

revision = '0020_control_requests'
down_revision = '0019_task_runtimes'
branch_labels = None
depends_on = None


def upgrade():
    from nndeploy.server.control import receipts
    receipts.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError('Automatic downgrade would remove control idempotency history')
