"""Public Server API. Importing this package does not initialize the editor."""

__all__ = ["Server", "NnDeployServer", "TaskQueue", "main"]


def __getattr__(name):
    if name in ("Server", "NnDeployServer"):
        from .server import Server
        return Server
    if name == "TaskQueue":
        from .task_queue import TaskQueue
        return TaskQueue
    if name == "main":
        from .app import main
        return main
    raise AttributeError(name)
