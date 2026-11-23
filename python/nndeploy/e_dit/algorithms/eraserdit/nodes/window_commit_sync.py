"""EraserDiT erase – window commit sync stage.

The commit handshake is windowing-runtime behaviour, not model behaviour, so the
EraserDiT pipeline reuses the shared implementation unchanged.
"""

from __future__ import annotations

from nndeploy.e_dit.nodes.base import operation

from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.nodes.base import StateNode
from nndeploy.e_dit.algorithms.eraserdit.video.commit_sync import synchronize_window_commit


class EraserDiTEraseWindowCommitSyncNode(StateNode):
    @operation
    def run(self, batch: Req, server_args: ServerArgs) -> Req:
        return synchronize_window_commit(batch, server_args)
