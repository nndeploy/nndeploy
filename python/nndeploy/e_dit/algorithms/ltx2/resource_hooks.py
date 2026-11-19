"""Two-stage adapter switching and optional upsampler ownership."""
from nndeploy.e_dit.adapters.lora import LoRAManager
from nndeploy.e_dit.loader.component_loaders.component_loader import PipelineComponentLoader

class LTX2Controller:
    def __init__(self, resources, two_stage):
        self.resources = resources
        self.two_stage = two_stage
        self.pipeline_name = 'LTX2TwoStagePipeline' if two_stage else 'LTX2Pipeline'
        model = resources.modules['transformer']
        self.lora = LoRAManager(model, name_mapping=getattr(model,'param_names_mapping',{}))
        self.base = False
        self.phase = None

    def open(self):
        args = self.resources.config
        if args.lora_path:
            self.lora.load('base',args.lora_path)
            self.base = True
        if self.two_stage:
            module, usage = PipelineComponentLoader.load_component(
                'spatial_upsampler',args.component_paths['spatial_upsampler'],'diffusers',args)
            self.resources.modules['spatial_upsampler'] = module
            self.resources.memory_usages['spatial_upsampler'] = usage
            self.lora.load('distilled',args.component_paths['distilled_lora'])
        self.switch_lora_phase('stage1')

    def switch_lora_phase(self, phase):
        if phase not in ('stage1','stage2'):
            raise ValueError(f'Unknown LoRA phase {phase}')
        if phase == self.phase: return
        active = [('base',self.resources.config.lora_scale)] if self.base else []
        if phase == 'stage2': active.append(('distilled',1.0))
        self.lora.activate(active)
        self.phase = phase

    def close(self):
        self.lora.close()
        self.resources = None
