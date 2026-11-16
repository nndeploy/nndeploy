"""Distinguish confirmed initial loading failures from established configurations."""
from alembic import op
from sqlalchemy import Boolean, Column, inspect

revision = '0017_initial_readiness'
down_revision = '0016_shared_configs'
branch_labels = None
depends_on = None


def upgrade():
    columns = {column['name'] for column in inspect(op.get_bind()).get_columns('server_shared_configs')}
    # Existing rows may have served requests. Never infer first-load failure
    # merely because all replicas have exited or their heartbeats disappeared.
    for column in (Column('initialized', Boolean, nullable=False, server_default='1'),
                   Column('load_failed', Boolean, nullable=False, server_default='0')):
        if column.name not in columns:
            op.add_column('server_shared_configs', column)


def downgrade():
    raise RuntimeError('Automatic downgrade would remove shared initialization history')
