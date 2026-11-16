"""Serialize schema changes across processes sharing a database."""
from contextlib import contextmanager
import os
import threading

# Alembic's environment proxy is process-global even for different databases.
local_lock = threading.RLock()


@contextmanager
def schema_lock(db):
    with local_lock:
        if db.engine.dialect.name == 'sqlite':
            marker = None
            try:
                if db.db_path is not None:
                    marker = open(str(db.db_path) + '.migration.lock', 'a+b')
                    if os.name == 'nt':
                        import msvcrt
                        marker.write(b'1')
                        marker.flush()
                        marker.seek(0)
                        msvcrt.locking(marker.fileno(), msvcrt.LK_LOCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(marker, fcntl.LOCK_EX)
                with db.engine.begin() as connection:
                    yield connection
            finally:
                if marker:
                    marker.close()
        else:
            with db.engine.connect() as connection:
                postgres = db.engine.dialect.name == 'postgresql'
                if postgres:
                    connection.exec_driver_sql('SELECT pg_advisory_lock(7244885193601)')
                else:
                    result = connection.exec_driver_sql("SELECT GET_LOCK('nndeploy_schema_migration', 30)").scalar()
                    if result != 1:
                        raise RuntimeError('Timed out waiting for the database migration lock')
                connection.commit()
                try:
                    with connection.begin():
                        yield connection
                finally:
                    connection.rollback()
                    connection.exec_driver_sql('SELECT pg_advisory_unlock(7244885193601)' if postgres else
                                               "SELECT RELEASE_LOCK('nndeploy_schema_migration')")
                    connection.commit()
