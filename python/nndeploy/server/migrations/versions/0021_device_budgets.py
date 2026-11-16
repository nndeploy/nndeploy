"""Resident model budgets with physical exit ownership."""
from alembic import op

revision = '0021_device_budgets'
down_revision = '0020_control_requests'
branch_labels = None
depends_on = None


def upgrade():
    from nndeploy.server.devices import scopes, claims
    scopes.create(op.get_bind(), checkfirst=True)
    claims.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError('Automatic downgrade would discard resident model resource ownership')
