# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang multimodal_gen/runtime/entrypoints/openai.
"""Source callback payload contract, independent of delivery and persistence."""
import json
from typing import Any, Dict
from ..video.progress import read_videoedit_progress

def _job_error_message(job: Dict[str, Any]) -> str:
    error = job.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if message:
            return str(message)
    if isinstance(error, str):
        return error
    return "failed"


def current_progress(job: Dict[str, Any]) -> int:
    progress_payload = read_videoedit_progress(job.get("progress_path"))
    progress = progress_payload.get("progress", job.get("progress", 0))
    if job.get("status") == "completed":
        progress = 100
    return int(progress)


def callback_payload(
    task_id: str, job: Dict[str, Any]
) -> Dict[str, Any]:
    status = job.get("status")
    progress = current_progress(job)
    if status == "completed":
        gen_video_url = (
            job.get("output_object_key") or job.get("url") or job.get("file_path") or ""
        )
        duration = job.get("inference_time_s")
        if duration is None and job.get("completed_at") and job.get("created_at"):
            duration = max(0, int(job["completed_at"] - job["created_at"]))
        output = {"gen_video_url": gen_video_url}
        if duration is not None:
            output["duration"] = duration
        return {
            "status": "succeeded",
            "progress": 100,
            "reason": "",
            "output": json.dumps(output, ensure_ascii=False),
        }
    if status in ("failed", "cancelled", "expired"):
        return {
            "status": status,
            "progress": progress,
            "reason": _job_error_message(job),
            "output": "",
        }
    return {
        "status": "running",
        "progress": progress,
        "reason": "",
        "output": "",
    }



def task_response(record):
    """Project the common durable task onto the source VideoEdit job contract."""
    service = record.get('service') or {}
    outputs = record.get('outputs') or {}
    progress = record.get('_progress') or {}
    progress = progress.get('progress', 0)
    if isinstance(progress, dict):
        progress = int(float(progress.get('fraction', 0)) * 100)
    state = record['state']
    status = {'pending': 'queued', 'dispatched': 'running', 'running': 'running',
              'succeeded': 'completed', 'cancelled': 'cancelled', 'expired': 'expired', 'failed': 'failed'}[state]
    if state == 'pending' and service.get('contract') == 'gateway':
        status = 'dispatching'
    if record.get('cancel_requested') and state not in ('succeeded', 'failed', 'cancelled', 'expired'):
        status = 'cancelling'
    error = record.get('error')
    identity = service.get('external_id', record['task_id'])
    from urllib.parse import quote
    url = outputs.get('url')
    file_path = outputs.get('file_path')
    if not file_path and outputs.get('artifacts'):
        file_path = outputs['artifacts'][0]['file'].get('local_path')
    if not url and not file_path and outputs.get('artifacts'):
        url = f'/v1/videos/{quote(record["task_id"], safe="")}/content'
    callback = record.get('callback') or {}
    if state not in ('succeeded', 'failed', 'cancelled', 'expired') and record.get('progress_delivery'):
        callback = json.loads(record['progress_delivery'])
    return dict(id=identity, object='video', size='', seconds='', quality='standard', model=service.get('model') or 'videoedit',
        variant=service.get('variant', 'normal'), status=status, progress=100 if state == 'succeeded' else min(99, int(progress)),
        created_at=int(record['created_at']), started_at=record.get('started_at'), completed_at=int(record['finished_at']) if record.get('finished_at') else None,
        url=url, file_path=file_path, error=error, reason=_job_error_message({'error': error}) if error else None,
        inference_time_s=outputs.get('inference_time_s'), peak_memory_mb=outputs.get('peak_memory_mb'),
        output_object_key=outputs.get('output_object_key'), queue_position=None,
        callback_status={'delivered': 'succeeded'}.get(callback.get('state'), callback.get('state')),
        callback_error=callback.get('error'), callback_attempts=callback.get('attempts', 0))


def task_callback(record):
    return callback_payload(record['task_id'], task_response(record))
