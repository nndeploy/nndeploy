"""Resident VSR weights and serialized request execution."""
from pathlib import Path
from threading import Lock


class VSRResources:
    def __init__(self, config):
        self.config = config
        self.restorer = None
        self.lock = Lock()
        self.last_metrics = None

    def open(self):
        if self.restorer is not None:
            return
        config = self.config
        if not Path(config.wan_root, 'vae').is_dir():
            raise ValueError('wan_root must contain the base Wan VAE directory')
        from .model import VSRRestorer, _resolve_weights
        _resolve_weights(config.checkpoint_dir, config.wan_root)
        if config.tile_devices:
            from .parallel import ParallelVSRRestorer
            self.restorer = ParallelVSRRestorer(config.tile_devices, **config.model_kwargs())
        else:
            self.restorer = VSRRestorer.from_pretrained(device=config.device, **config.model_kwargs())

    def close(self):
        with self.lock:
            restorer, self.restorer = self.restorer, None
            if restorer is not None and hasattr(restorer, 'close'):
                restorer.close()

    def restore(self, video, output, options):
        from .geometry import parse_resolution, resize_to_long_edge
        from .stream import stream_restore
        from .video_io import probe_video
        from nndeploy.e_dit.video.artifacts import validate_output_paths
        from nndeploy.dag.execution import current_execution
        from nndeploy.e_dit.nodes.control import RequestCancelled
        import torch

        settings = self.config.request(options)
        video, output = Path(video), Path(output)
        if not video.is_file():
            raise ValueError('video must be a local file')
        if output.suffix.lower() != '.mp4':
            raise ValueError('VSR output must use .mp4')
        validate_output_paths([output], [video])
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('A VSR graph executes one request at a time')
        try:
            if self.restorer is None:
                raise RuntimeError('Initialize VSR before inference')
            self.last_metrics = None
            control = current_execution()
            def checkpoint():
                if control and control.cancelled():
                    raise RequestCancelled('VSR request cancelled')
            checkpoint()
            if control and control.publish:
                control.publish({'progress': {'phase': 'restore', 'fraction': 0.0}})
            fps, count, h, w = probe_video(video)
            target = settings.pop('target_resolution')
            long_edge = settings.pop('long_edge')
            h, w = parse_resolution(target) if target else resize_to_long_edge(h, w, long_edge)
            for key, attr in [('tile_t', 'tile_t'), ('tile_h', 'tile_h'), ('tile_w', 'tile_w'),
                              ('temporal_overlap', 't_overlap'), ('spatial_overlap', 's_overlap')]:
                setattr(self.restorer, attr, settings.pop(key))
            settings['color_samples'] = settings.pop('color_ref_samples')
            output.parent.mkdir(parents=True, exist_ok=True)
            with torch.inference_mode():
                written = stream_restore(self.restorer, video, output, target_h=h, target_w=w,
                                         fps=fps, total_frames=count, check_interrupt=checkpoint,
                                         **settings)
            self.last_metrics = {'frames_written': written, 'height': h, 'width': w, 'fps': fps}
            if control and control.publish:
                control.publish({'progress': {'phase': 'completed', 'fraction': 1.0}})
            return str(output)
        finally:
            self.lock.release()
