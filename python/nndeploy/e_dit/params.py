"""Serializable instance parameters; no models or process handles live here."""
from copy import deepcopy
import json

from nndeploy.base import Param, Status


class DiTParams(Param):
    def __init__(self, model_path='', *, model='eraserdit', pipeline=None, **options):
        super().__init__()
        from nndeploy.e_dit.registry import model_names
        if model not in model_names():
            raise ValueError(f'Unsupported model: {model}')
        self.model = model
        self.model_path = str(model_path)
        self.pipeline = deepcopy(pipeline or {})
        self.options = deepcopy(options)
        # Validate JSON at the boundary, before loading any device resources.
        self.serialize()

    def to_dict(self):
        return dict(model=self.model, model_path=self.model_path,
                    pipeline=deepcopy(self.pipeline), **deepcopy(self.options))

    def serialize(self):
        return json.dumps(self.to_dict(), allow_nan=False)

    def deserialize(self, value):
        data = json.loads(value) if isinstance(value, str) else value
        other = type(self)(**data)
        self.model, self.model_path = other.model, other.model_path
        self.pipeline, self.options = other.pipeline, other.options
        return Status.ok()

    def save_file(self, path):
        from pathlib import Path
        Path(path).write_text(self.serialize() + '\n')
        return Status.ok()

    def load_file(self, path):
        from pathlib import Path
        return self.deserialize(Path(path).read_text())

    def set(self, key, value):
        data = self.to_dict()
        data[key] = value
        return self.deserialize(data)

    def get(self, key):
        return self.to_dict()[key]

    def build_config(self):
        from nndeploy.e_dit.registry import get_model
        return get_model(self.model).build_config(self)
