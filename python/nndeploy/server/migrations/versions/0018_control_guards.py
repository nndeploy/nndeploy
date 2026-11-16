"""Associate unfinished shared changes with physical runtime exit evidence."""
from alembic import op
from sqlalchemy import Column, String, inspect

revision = '0018_control_guards'
down_revision = '0017_initial_readiness'
branch_labels = None
depends_on = None


def upgrade():
    for table, column in (('server_shared_entries', 'pending_holder'),
                          ('server_shared_workers', 'owner_token')):
        if column not in {value['name'] for value in inspect(op.get_bind()).get_columns(table)}:
            op.add_column(table, Column(column, String(32)))


def downgrade():
    raise RuntimeError('Automatic downgrade would remove shared runtime ownership evidence')
