"""Native EraserDiT compute nodes and their DAG registrations."""

from nndeploy.e_dit.algorithms.eraserdit.nodes.condition_encoding import (
    EraserDiTEraseConditionEncodingNode,
)
from nndeploy.e_dit.algorithms.eraserdit.nodes.decoding import (
    EraserDiTEraseDecodingNode,
)
from nndeploy.e_dit.algorithms.eraserdit.nodes.denoising import (
    EraserDiTEraseDenoisingNode,
)
from nndeploy.e_dit.algorithms.eraserdit.nodes.latent_preparation import (
    EraserDiTEraseLatentPreparationNode,
)
from nndeploy.e_dit.algorithms.eraserdit.nodes.preprocess import (
    EraserDiTErasePreprocessNode,
)
from nndeploy.e_dit.algorithms.eraserdit.nodes.text_encoding import (
    EraserDiTEraseTextEncodingNode,
)
from nndeploy.e_dit.algorithms.eraserdit.nodes.timestep_preparation import (
    EraserDiTEraseTimestepPreparationNode,
)
from nndeploy.e_dit.algorithms.eraserdit.nodes.window_commit_sync import (
    EraserDiTEraseWindowCommitSyncNode,
)
from nndeploy.e_dit.algorithms.eraserdit.nodes.window_postprocess import (
    EraserDiTEraseWindowPostprocessNode,
)
from nndeploy.e_dit.algorithms.eraserdit.nodes.window_validation import (
    EraserDiTEraseWindowValidationNode,
)

__all__ = [
    "EraserDiTEraseWindowValidationNode",
    "EraserDiTEraseTextEncodingNode",
    "EraserDiTErasePreprocessNode",
    "EraserDiTEraseConditionEncodingNode",
    "EraserDiTEraseLatentPreparationNode",
    "EraserDiTEraseTimestepPreparationNode",
    "EraserDiTEraseDenoisingNode",
    "EraserDiTEraseDecodingNode",
    "EraserDiTEraseWindowPostprocessNode",
    "EraserDiTEraseWindowCommitSyncNode",
]

from nndeploy.e_dit.nodes.graph import register_type
for _name in __all__:
    if _name != "EraserDiTEraseDenoisingNode":
        register_type(globals()[_name])
