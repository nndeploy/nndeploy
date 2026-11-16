"""Resolve managed resources in the editor's private execution copy."""
from pathlib import Path

from .resources import prepare_workflow, workflow_file_refs


def prepare_worker_inputs(payload, options, resources, *, retain=None):
    workflow = payload['graph_json']
    if not workflow_file_refs(workflow):
        return workflow
    if options is None:
        raise ValueError('Managed editor inputs require Server storage')
    from .config import ServerConfig
    from .db import DB
    from .storage import LocalStorage
    db = DB(options['database_url'], timeout_seconds=options['timeout_seconds'])
    storage = None
    try:
        lease = None
        if retain is not None:
            from .assets import AssetLease
            lease = AssetLease(Path(options['root']).parent, db)
            retain(lease)
        storage = LocalStorage(options['root'], db, max_bytes=options['max_bytes'], config=options['config'])
        # Admission already protects these files until task history expires.
        # Preserve local paths and the submitted JSON; only IDs in the Worker
        # copy become paths before native Graph deserialization.
        return prepare_workflow(workflow, ServerConfig(name='editor'), Path(options['root']).parent,
                                resources, storage=storage, _copy_local=False, lease=lease)
    finally:
        if storage is not None:
            storage.close()
        db.close()
