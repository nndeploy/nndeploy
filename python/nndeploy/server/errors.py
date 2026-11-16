"""Errors shared by Python callers and HTTP routes."""

class ServerError(RuntimeError):
    def __init__(self, status_code, code, message, *, task_id=None, retryable=False, status_url=None, accepted=None,
                 retry_after=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.task_id = task_id
        self.retryable = retryable
        self.status_url = status_url
        self.accepted = accepted
        self.retry_after = retry_after

    def response(self):
        detail = {"code": self.code, "message": str(self),
                  "task_id": self.task_id, "retryable": self.retryable, "status_code": self.status_code}
        if self.status_url is not None:
            detail['status_url'] = self.status_url
        if self.accepted is not None:
            detail['accepted'] = self.accepted
        return {"error": detail}

    @classmethod
    def from_task(cls, record):
        error = record.get('error') or {}
        status = {'failed': 500, 'cancelled': 409, 'expired': 504}.get(record['state'], 409)
        return cls(error.get('status_code', status), error.get('code', 'task_in_progress'),
                   error.get('message', 'Task is owned by another Server; use its status URL'),
                   task_id=record['task_id'], retryable=error.get('retryable', False))
