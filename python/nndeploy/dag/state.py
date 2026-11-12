"""Typed request state backed by the same storage used by native DAG nodes."""

import dataclasses
import weakref

from nndeploy import _nndeploy_internal as _C


class State:
    """Subclass with @dataclass and access declared fields as attributes.

    Assignment is immediate. Objects (including tensors and lists) are shared,
    not copied. Order conflicting mutations using graph control dependencies.
    """

    def __new__(cls, *args, **kwargs):
        if cls is not State:
            if not dataclasses.is_dataclass(cls):
                raise TypeError("State subclasses must use @dataclass")
            if cls.__dataclass_params__.frozen:
                raise TypeError("State must be mutable (frozen=True is unsupported)")
            for field in dataclasses.fields(cls):
                if field.name.startswith("_") or hasattr(State, field.name):
                    raise TypeError(f"Reserved State field: {field.name}")
        instance = super().__new__(cls)
        storage = _C.dag._StateStorage()
        object.__setattr__(instance, "_native_state", storage)
        storage._owner = weakref.ref(instance)
        # init=False fields are not assigned by dataclass's generated __init__.
        if cls is not State:
            for field in dataclasses.fields(cls):
                if not field.init and field.default is not dataclasses.MISSING:
                    storage.set(field.name, field.default)
        return instance

    @classmethod
    def _from_native(cls, storage):
        instance = object.__new__(State)
        object.__setattr__(instance, "_native_state", storage)
        storage._owner = weakref.ref(instance)
        return instance

    def __getattribute__(self, name):
        if not name.startswith("_"):
            cls = type(self)
            declared = cls is not State and any(
                field.name == name for field in dataclasses.fields(cls)
            )
            storage = object.__getattribute__(self, "_native_state")
            if declared or (cls is State and storage.contains(name)):
                try:
                    return storage.get(name)
                except KeyError:
                    raise AttributeError(f"State field {name!r} is not initialized") from None
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        cls = type(self)
        storage = object.__getattribute__(self, "_native_state")
        declared = storage.contains(name) if cls is State else any(
            field.name == name for field in dataclasses.fields(cls))
        if name.startswith("_") or not declared:
            raise AttributeError(f"Undeclared or reserved State field: {name}")
        storage.set(name, value)

    def __delattr__(self, name):
        raise AttributeError("State fields cannot be deleted")
