"""Serialize identity task admission across Server replicas."""
import hashlib
import json

from alembic import op
from sqlalchemy import Column, MetaData, String, Table, bindparam, inspect, select, update

revision = '0026_subject_task_quotas'
down_revision = '0025_editor_tasks'
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    if 'owner_key' not in {c['name'] for c in inspect(connection).get_columns('server_tasks')}:
        op.add_column('server_tasks', Column('owner_key', String(64)))
    tasks = Table('server_tasks', MetaData(), Column('task_id', String(64)),
                  Column('owner', String(128)), Column('owner_key', String(64)))
    # Resumable in bounded batches, including tasks accepted before quotas were
    # configured. Do not silently give old live tasks free capacity on upgrade.
    while True:
        rows = connection.execute(select(tasks.c.task_id, tasks.c.owner).where(
            tasks.c.owner_key.is_(None)).order_by(tasks.c.task_id).limit(1000)).all()
        if not rows:
            break
        connection.execute(update(tasks).where(tasks.c.task_id == bindparam('_task')).values(owner_key=bindparam('_key')),
            [{'_task': row.task_id, '_key': hashlib.sha256(json.dumps(row.owner, ensure_ascii=False).encode()).hexdigest()}
             for row in rows])
    if 'ix_server_tasks_owner_state' not in {i['name'] for i in inspect(connection).get_indexes('server_tasks')}:
        op.create_index('ix_server_tasks_owner_state', 'server_tasks', ['owner_key', 'state'])
    Table('server_subject_locks', MetaData(), Column('subject_key', String(64), primary_key=True)).create(
        connection, checkfirst=True)


def downgrade():
    raise RuntimeError('Automatic downgrade would remove identity task admission coordination')
