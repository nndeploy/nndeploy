# SPDX-License-Identifier: Apache-2.0
"""VSR model components: the one-step restoration pipeline.

Port of ``infer/models/stage3.py``.

This is **not** a sampling diffusion pipeline. Each window gets exactly one DiT
forward at a fixed timestep with a zero text condition, and the latent is
updated by subtracting the predicted velocity::

    z_lq = (VAE.encode(x).mode() - mu) / sigma
    v    = DiT(z_lq, t=1000, encoder_hidden_states=zeros)
    x_hat = VAE.decode(z_lq - v)

No scheduler, no CFG, no noise, no text encoder, and no per-window randomness --
which is why the reference is reproducible run to run and why caching schemes
such as TeaCache or Cache-DiT have nothing to cache here
(``requirements.md`` §5.2).

Phase 1 deliberately uses the diffusers classes rather than SGLang's own Wan
implementations, so the numerics stay comparable to the reference; swapping them
in is phase 2 (``requirements.md`` §1).
"""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext

import torch
from torch import nn

#: The rectified-flow endpoint the reference evaluates at. It is a *timestep
#: condition*, not an iteration count and not a frame index.
FM_TIMESTEP = 1000.0


class VSRAutoencoder(nn.Module):
    """Base Wan2.2 VAE encoder plus an optional fine-tuned decoder.

    Stage-3 training only writes ``vae_decoder.pt``; the encoder stays frozen at
    the base Wan2.2 weights. With ``--use_ema`` the run also writes
    ``vae_decoder_ema.pt``, and inference prefers it so the EMA'd DiT is paired
    with the matching EMA'd decoder.

    The per-channel latent normalisation mirrors the training code so the DiT
    sees the same latent distribution. Encoding uses ``.mode()`` rather than
    ``.sample()``: deterministic, and it is what the reference measures against.
    """

    def __init__(self, vae, decoder_state_path: str | None = None):
        super().__init__()
        self.vae = vae
        if decoder_state_path is not None:
            state = torch.load(str(decoder_state_path), map_location="cpu")
            self.vae.decoder.load_state_dict(state)
        mean = torch.as_tensor(vae.config.latents_mean, dtype=torch.float32)
        std = torch.as_tensor(vae.config.latents_std, dtype=torch.float32)
        self.register_buffer("latents_mean", mean.view(1, -1, 1, 1, 1))
        self.register_buffer("latents_std", std.view(1, -1, 1, 1, 1))

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """``[B, C, T, H, W]`` in ``[-1, 1]`` -> normalised latents (deterministic)."""
        vae_dtype = next(self.vae.parameters()).dtype
        latent = self.vae.encode(frames.to(vae_dtype)).latent_dist.mode()
        return (latent - self.latents_mean.to(latent)) / self.latents_std.to(latent)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Normalised latents -> ``[B, C, T, H, W]`` in ``[-1, 1]``.

        Note ``.sample`` here is the output field of the decoder result object,
        not the ``DiagonalGaussianDistribution.sample()`` method.
        """
        vae_dtype = next(self.vae.parameters()).dtype
        latent = latent.to(vae_dtype)
        latent = latent * self.latents_std.to(latent) + self.latents_mean.to(latent)
        return self.vae.decode(latent).sample


def _resolve_weights(checkpoint_dir, wan_root):
    """Pick the transformer and decoder paths, applying the reference's rules.

    EMA DiT must be paired with ``vae_decoder_ema.pt``; mixing an EMA DiT with
    the online decoder would silently combine weights from two different
    training runs. When the matching decoder is missing the reference keeps the
    *base* decoder -- it does not fall back to the other variant.
    """
    from pathlib import Path

    ckpt = Path(checkpoint_dir)
    ema_path = ckpt / "transformer_ema"
    online_path = ckpt / "transformer"
    decoder_ema = ckpt / "vae_decoder_ema.pt"
    decoder_online = ckpt / "vae_decoder.pt"

    if ema_path.exists():
        transformer_path = ema_path
        decoder_path = str(decoder_ema) if decoder_ema.exists() else None
        variant = "ema"
    elif online_path.exists():
        transformer_path = online_path
        decoder_path = str(decoder_online) if decoder_online.exists() else None
        variant = "online"
    else:
        raise FileNotFoundError(
            f"No transformer weights in checkpoint dir: {ckpt} "
            f"(looked for transformer_ema/ and transformer/)"
        )
    return transformer_path, decoder_path, variant


class VSRRestorer(nn.Module):
    """One-step restoration of a single pixel window.

    The window is ``[B, C, tile_t, tile_h, tile_w]`` in ``[-1, 1]`` and the
    result has the same shape. Tiling, blending and streaming live in
    ``blending`` / ``stream``; this class only owns the model call.
    """

    def __init__(
        self,
        vae: VSRAutoencoder,
        dit,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        tile_t: int = 33,
        tile_h: int = 320,
        tile_w: int = 640,
        t_overlap: int = 5,
        s_overlap: int = 32,
        cudnn_benchmark: bool = False,
        channels_last_3d: bool = False,
        compile_decoder: bool = False,
        compile_encoder: bool = False,
        decoder_implicit_padding: bool = False,
        cache_dit_condition: bool = False,
        vae_cpu_offload: bool = False,
        dit_cpu_offload: bool = False,
        dit_layerwise_offload: bool = False,
        dit_offload_prefetch_size: float = 0.0,
        pin_cpu_memory: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.tile_t = tile_t
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.t_overlap = t_overlap
        self.s_overlap = s_overlap
        self.cudnn_benchmark = cudnn_benchmark

        if dit_cpu_offload and dit_layerwise_offload:
            raise ValueError(
                "DiT whole-module and layerwise offload are mutually exclusive"
            )
        if (
            not math.isfinite(dit_offload_prefetch_size)
            or dit_offload_prefetch_size < 0
        ):
            raise ValueError(
                "dit_offload_prefetch_size must be finite and non-negative"
            )
        if dit_layerwise_offload and self.device.type != "cuda":
            raise ValueError("VSR layerwise offload requires CUDA")
        self.vae_cpu_offload = vae_cpu_offload
        self.dit_cpu_offload = dit_cpu_offload
        self.dit_offload_manager = None
        self.vae = vae.to("cpu" if vae_cpu_offload else device, dtype=dtype).eval()
        if channels_last_3d:
            torch.nn.utils.convert_conv3d_weight_memory_format(
                self.vae, torch.channels_last_3d
            )
        self.dit = dit.to(
            "cpu" if dit_cpu_offload or dit_layerwise_offload else device, dtype=dtype
        ).eval()
        if dit_layerwise_offload:
            from nndeploy.e_dit.memory.backends.layerwise_offload import (
                LayerwiseOffloadManager,
            )

            num_layers = len(self.dit.blocks)
            if not num_layers:
                raise ValueError("DiT layerwise offload requires nonempty blocks")
            prefetch = (
                1 + round(dit_offload_prefetch_size * (num_layers - 1))
                if dit_offload_prefetch_size < 1
                else int(dit_offload_prefetch_size)
            )
            # Initialize directly from CPU weights to avoid a full-DiT GPU peak.
            with torch.cuda.device(self.device):
                self.dit_offload_manager = LayerwiseOffloadManager(
                    self.dit,
                    layers_attr_str="blocks",
                    num_layers=num_layers,
                    enabled=True,
                    pin_cpu_memory=pin_cpu_memory,
                    prefetch_size=prefetch,
                )
                self.dit_offload_manager.release_all()
                for name, child in self.dit.named_children():
                    if name != "blocks":
                        child.to(self.device)
                for tensor in list(self.dit.parameters(recurse=False)) + list(
                    self.dit.buffers(recurse=False)
                ):
                    tensor.data = tensor.data.to(self.device)
        if cache_dit_condition:
            from nndeploy.e_dit.algorithms.vsr.condition_cache import (
                cache_fixed_vsr_condition,
            )

            cache_fixed_vsr_condition(self.dit)
        if decoder_implicit_padding:
            from nndeploy.e_dit.algorithms.vsr.compile import (
                use_implicit_decoder_padding,
            )

            use_implicit_decoder_padding(self.vae.vae)
        if compile_encoder:
            from nndeploy.e_dit.algorithms.vsr.compile import (
                compile_encoder as compile_vae_encoder,
            )

            compile_vae_encoder(self.vae.vae, offload=vae_cpu_offload)
        if compile_decoder:
            from nndeploy.e_dit.algorithms.vsr.compile import (
                compile_decoder as compile_vae_decoder,
            )

            compile_vae_decoder(self.vae.vae, offload=vae_cpu_offload)

        # Zero text conditioning. Deliberately a zero tensor and not the
        # embedding of an empty string: the reference loads no text encoder at
        # all. See requirements.md §7-3.
        text_dim = self.dit.config.text_dim
        self.register_buffer(
            "empty_prompt", torch.zeros(1, 1, text_dim, dtype=dtype, device=self.device)
        )

    def close(self):
        """Detach offload hooks so model/manager cycles do not retain GPU memory."""
        manager = self.dit_offload_manager
        if manager is not None:
            try:
                manager.release_all()
            finally:
                manager.remove_forward_hooks()
                self.dit_offload_manager = None

    @torch.no_grad()
    def restore_window(self, window: torch.Tensor) -> torch.Tensor:
        """Restore one pixel window using this model's convolution settings."""
        # Backend flags are process-global: restore the caller's setting even
        # on failure. Like the VAE's mutable causal cache, this requires serial
        # model execution within a worker process.
        previous = torch.backends.cudnn.benchmark
        try:
            torch.backends.cudnn.benchmark = self.cudnn_benchmark
            with (
                torch.cuda.device(self.device)
                if self.device.type == "cuda"
                else nullcontext()
            ):
                return self._restore_window(window)
        finally:
            torch.backends.cudnn.benchmark = previous

    @contextmanager
    def _on_device(self, module, offload=False, manager=None):
        try:
            if offload:
                module.to(self.device)
            yield
        finally:
            if offload or manager is not None:
                from nndeploy.e_dit.algorithms.vsr.condition_cache import (
                    clear_fixed_vsr_condition,
                )

                clear_fixed_vsr_condition(module)
                # Wan keeps temporal features outside registered buffers.
                if module is self.vae and hasattr(self.vae.vae, "clear_cache"):
                    self.vae.vae.clear_cache()
                if manager is not None:
                    manager.release_all()
                if offload:
                    module.to("cpu")

    def _restore_window(self, window: torch.Tensor) -> torch.Tensor:
        """FM one-step restore of ``[B, C, tile_t, H, W]``.

        ``z_hq = z_lq - v(z_lq, t=1000)`` -- the rectified-flow sigma=1 endpoint.
        """
        window = window.to(self.device, dtype=self.dtype)
        B = window.shape[0]

        with self._on_device(self.vae, self.vae_cpu_offload):
            z_lq = self.vae.encode(window)
        t = torch.full((B,), FM_TIMESTEP, device=self.device, dtype=torch.float32)
        prompt = self.empty_prompt.expand(B, -1, -1)
        with self._on_device(self.dit, self.dit_cpu_offload, self.dit_offload_manager):
            velocity = self.dit(
                hidden_states=z_lq,
                timestep=t,
                encoder_hidden_states=prompt,
                return_dict=False,
            )[0]
        z_hq = z_lq - velocity
        with self._on_device(self.vae, self.vae_cpu_offload):
            return self.vae.decode(z_hq)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str,
        wan_root: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        tile_t: int = 33,
        tile_h: int = 320,
        tile_w: int = 640,
        t_overlap: int = 5,
        s_overlap: int = 32,
        verbose: bool = True,
        cudnn_benchmark: bool = False,
        channels_last_3d: bool = False,
        compile_decoder: bool = False,
        compile_encoder: bool = False,
        decoder_implicit_padding: bool = False,
        cache_dit_condition: bool = False,
        vae_cpu_offload: bool = False,
        dit_cpu_offload: bool = False,
        dit_layerwise_offload: bool = False,
        dit_offload_prefetch_size: float = 0.0,
        pin_cpu_memory: bool = True,
    ) -> VSRRestorer:
        """Load a Stage-3 checkpoint.

        Args:
            checkpoint_dir: holds ``transformer_ema/`` + ``vae_decoder_ema.pt``,
                or the non-EMA ``transformer/`` + ``vae_decoder.pt``.
            wan_root: base Wan2.2-TI2V-5B-Diffusers directory. Only ``vae/`` is
                read from it -- the DiT always comes from ``checkpoint_dir``.
        """
        from diffusers import AutoencoderKLWan, WanTransformer3DModel

        transformer_path, decoder_path, variant = _resolve_weights(
            checkpoint_dir, wan_root
        )
        if verbose:
            print(f"[vsr] transformer: {transformer_path} ({variant})")
            if decoder_path:
                print(f"[vsr] fine-tuned VAE decoder: {decoder_path}")
            else:
                print("[vsr] no fine-tuned VAE decoder found; using the base decoder")

        base_vae = AutoencoderKLWan.from_pretrained(wan_root, subfolder="vae")
        vae = VSRAutoencoder(base_vae, decoder_state_path=decoder_path)
        dit = WanTransformer3DModel.from_pretrained(str(transformer_path))

        return cls(
            vae=vae,
            dit=dit,
            device=device,
            dtype=dtype,
            tile_t=tile_t,
            tile_h=tile_h,
            tile_w=tile_w,
            t_overlap=t_overlap,
            s_overlap=s_overlap,
            cudnn_benchmark=cudnn_benchmark,
            channels_last_3d=channels_last_3d,
            compile_decoder=compile_decoder,
            compile_encoder=compile_encoder,
            decoder_implicit_padding=decoder_implicit_padding,
            cache_dit_condition=cache_dit_condition,
            vae_cpu_offload=vae_cpu_offload,
            dit_cpu_offload=dit_cpu_offload,
            dit_layerwise_offload=dit_layerwise_offload,
            dit_offload_prefetch_size=dit_offload_prefetch_size,
            pin_cpu_memory=pin_cpu_memory,
        )
