"""Track temporary execution memory and declared runtime headroom."""
from alembic import op
from sqlalchemy import BigInteger, Column, String, inspect
from sqlalchemy.dialects.mysql import VARCHAR

revision = '0023_temporary_memory'
down_revision = '0022_device_execution'
branch_labels = None
depends_on = None


def upgrade():
    for table, name in (('server_device_claims', 'peak_memory_bytes'),
                        ('server_device_run_devices', 'memory_bytes')):
        existing = {column['name'] for column in inspect(op.get_bind()).get_columns(table)}
        if name not in existing:
            op.add_column(table, Column(name, BigInteger, nullable=False, server_default='0'))
    if op.get_bind().dialect.name == 'mysql':
        # Policies use exact Python identifiers regardless of database defaults.
        # Preserve that meaning in primary keys and grouped memory accounting.
        for table, name in (('server_device_claims', 'device'), ('server_device_claims', 'name'),
                            ('server_device_run_devices', 'device'), ('server_device_runs', 'name')):
            op.alter_column(table, name, existing_type=String(128), existing_nullable=False,
                            type_=VARCHAR(128, charset='utf8mb4', collation='utf8mb4_bin'))


def downgrade():
    raise RuntimeError('Automatic downgrade would remove active device memory ownership')
