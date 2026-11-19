# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang VSR/OpenAI video interfaces; see ../LICENSE.SGLang.
from typing import Dict, Any
from .protocol import _job_reason

def _build_video_callback_payload(
    video_id: str, job: Dict[str, Any]
) -> Dict[str, Any]:
    payload = {
        "id": video_id,
        "object": job.get("object", "video"),
        "model": job.get("model"),
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "created_at": job.get("created_at"),
        "completed_at": job.get("completed_at"),
        "file_path": job.get("file_path"),
        "url": job.get("url"),
        "error": job.get("error"),
        "reason": _job_reason(job),
    }
    for key in ("peak_memory_mb", "inference_time_s"):
        if key in job:
            payload[key] = job[key]
    return payload

def task_response(record):
    """Expose distinct failure, cancellation and deadline outcomes."""
    service = record.get('service') or {}
    outputs = record.get('outputs') or {}
    detail = record.get('_progress') or {}
    state = record['state']
    terminal = state in ('succeeded', 'failed', 'cancelled', 'expired')
    status = {'pending': 'queued', 'dispatched': 'running', 'running': 'running',
              'succeeded': 'completed', 'failed': 'failed', 'cancelled': 'cancelled', 'expired': 'expired'}[state]
    if record.get('cancel_requested') and not terminal:
        status = 'cancelling'
    error = record.get('error')
    url = outputs.get('url')
    artifacts = outputs.get('artifacts') or []
    path = artifacts[0]['file'].get('local_path') if artifacts and not url else None
    if artifacts and not path and not url:
        from urllib.parse import quote
        url = '/v1/videos/' + quote(record['task_id'], safe='') + '/content'
    callback = record.get('callback') or {}
    progress = detail.get('progress', 0)
    if isinstance(progress, dict):
        progress = float(progress.get('fraction', 0)) * 100
    return dict(id=service.get('external_id', record['task_id']), object='video', model='vsr',
        status=status, progress=100 if state == 'succeeded' else min(99, max(0, int(progress))),
        created_at=int(record['created_at']), size=outputs.get('size', detail.get('size', '')),
        seconds=outputs.get('seconds', detail.get('seconds', '4')), file_path=path, url=url,
        completed_at=int(record['finished_at']) if terminal and record.get('finished_at') else None,
        error=error, reason=error.get('message') if error else None,
        inference_time_s=outputs.get('inference_time_s'), peak_memory_mb=outputs.get('peak_memory_mb'),
        callback_status={'delivered': 'succeeded'}.get(callback.get('state'), callback.get('state')),
        callback_error=callback.get('error'),
        callback_attempts=callback.get('attempts', 0) if callback.get('url') else None)


def task_callback(record):
    job = task_response(record)
    for key in ('peak_memory_mb', 'inference_time_s'):
        if job.get(key) is None:
            job.pop(key, None)
    return _build_video_callback_payload(job['id'], job)
