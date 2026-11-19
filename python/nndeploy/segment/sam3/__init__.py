# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Original SAM3 image/video inference, distributed under the SAM License.

Model dependencies are loaded only when a builder or processor is requested.
"""

from importlib import import_module

__all__ = [
    "build_sam3_image_model",
    "build_sam3_video_model",
    "build_sam3_video_predictor",
    "Sam3Processor",
    "ModelConfig",
    "RuntimeOwner",
    "SAM3ImageConceptGraph",
    "SAM3ImageInteractiveGraph",
    "SAM3VideoGraph",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name == "ModelConfig":
        module_name = ".types"
    elif name == "RuntimeOwner":
        module_name = ".runtime"
    elif name.endswith("Graph"):
        module_name = ".graphs"
    else:
        module_name = ".model.sam3_image_processor" if name == "Sam3Processor" else ".model_builder"
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
