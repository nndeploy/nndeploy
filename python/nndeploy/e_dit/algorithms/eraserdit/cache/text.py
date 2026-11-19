"""Attention projection binding; lifetime is owned by the algorithm request."""
from nndeploy.e_dit.cache.projection import ProjectionCache


class EraserDiTTextCache(ProjectionCache):
    def key_value(self, attn, encoder_hidden_states):
        return self.get_or_compute(id(attn), lambda: (
            attn.norm_k(attn.to_k(encoder_hidden_states)), attn.to_v(encoder_hidden_states)))
