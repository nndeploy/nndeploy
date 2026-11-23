"""Translate model progress and safe cancellation boundaries to DAG control."""
class CancellationProbe:
    def __init__(self, cancelled):
        self.cancelled = cancelled

    @property
    def requested(self):
        return self.cancelled()

    def touch(self):
        pass


class ProgressSink:
    def __init__(self, publish):
        self.publish = publish
        self.location = {}
        self.completed, self.total = 0, 1
        self.fraction = 0.

    def stop(self): pass
    def hide_denoise(self): pass
    def set_pipeline_meta(self, meta): pass

    def add_pipeline_task(self, total):
        self.total = max(1, int(total))

    def update_pipeline(self, completed, total, **location):
        self.completed, self.total = completed, max(1, total)
        self.location = location
        self._emit('window', self.completed / self.total)

    def reset_denoise_task(self, **location):
        self.location = location
        self._emit('denoise', self.completed / self.total)

    def update_denoise(self, step_index, total_steps, **extra):
        fraction = (self.completed + .9 * (step_index + 1) / max(1, total_steps)) / self.total
        self._emit('denoise', fraction, step=step_index + 1, total_steps=total_steps)

    def stage(self, name, status):
        self._emit('stage', self.fraction, stage=name, status=status)

    def complete(self):
        self._emit('completed', 1.)

    def _emit(self, phase, fraction, **extra):
        if self.fraction >= 1.:
            return
        # Window completion still precedes encoder flush and output retrieval.
        ceiling = 1. if phase == 'completed' else .99
        self.fraction = min(ceiling, max(self.fraction, fraction))
        self.publish({'progress': {'phase': phase, 'fraction': self.fraction,
                                   **self.location, **extra}})
