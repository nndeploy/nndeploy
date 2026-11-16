"""Durable HTTP intent, authenticated identity and observed response outcome."""
from alembic import op

revision = '0028_http_audit'
down_revision = '0027_subject_file_quotas'
branch_labels = None
depends_on = None


def upgrade():
    from nndeploy.server.audit import records
    records.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError('Automatic downgrade would discard audit records')
