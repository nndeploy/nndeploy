"""Device execution slots belong to existing tasks and physical runtimes."""
from alembic import op

revision = '0022_device_execution'
down_revision = '0021_device_budgets'
branch_labels = None
depends_on = None


def upgrade():
    from nndeploy.server.devices import runs, run_devices
    runs.create(op.get_bind(), checkfirst=True)
    run_devices.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError('Automatic downgrade would discard active device execution ownership')
