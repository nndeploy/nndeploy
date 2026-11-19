"""Explicit owner for shared image/video resources; no process-global cache."""

import threading
from ..modules import Modules
from ..types import SAM3Error


class RuntimeOwner:
    def __init__(self, config, result_dir, modules=None):
        self.config = config
        self.result_dir = result_dir
        self.modules = modules or Modules()
        self._image = self._video = None
        self._closed = False
        self._lock = threading.RLock()

    @property
    def image(self):
        with self._lock:
            if self._closed:
                raise SAM3Error("runtime_closed", "Owner has been closed")
            if self._image is None:
                from .image import ImageRuntime
                self._image = ImageRuntime(self.config, self.modules)
            return self._image

    @property
    def video(self):
        with self._lock:
            if self._closed:
                raise SAM3Error("runtime_closed", "Owner has been closed")
            if self._video is None:
                from .video import VideoRuntime
                self._video = VideoRuntime(self.config, self.result_dir, self.modules)
            return self._video

    def close(self):
        with self._lock:
            for runtime in (self._image, self._video):
                if runtime is not None:
                    runtime.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
