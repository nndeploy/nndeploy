
from typing import List, Optional, Tuple, Union

# 🤗 Diffusers: Supported main tasks, representative pipelines, and models (categorized by function for quick reference).
# Pipelines provide end-to-end inference and automatically load required components. The following list is consistent with the official pipelines list.

# 🖼️ Image Generation and Processing

# Text-to-Image
supported_text2image_models = [
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "stable-diffusion-xl-base-1.0/stable-diffusion-xl-base-1.0",
    "stable-diffusion-xl-refiner-1.0/stable-diffusion-xl-refiner-1.0",
    "sdxl-turbo/sdxl-turbo",
    "sd-turbo/sd-turbo",
    "stable-diffusion-2-1/stable-diffusion-2-1",
    # Stable Diffusion Series
    "runwayml/stable-diffusion-v1-5",                # Stable Diffusion 1.5 (StableDiffusionPipeline)
    "stabilityai/stable-diffusion-xl-base-1.0",      # Stable Diffusion XL Base (StableDiffusionXLPipeline)
    "stabilityai/stable-diffusion-xl-refiner-1.0",   # Stable Diffusion XL Refiner
    "stabilityai/sdxl-turbo",                        # Stable Diffusion XL Turbo (StableDiffusionXLInstantIDPipeline)
    # Kandinsky Series
    "kandinsky-community/kandinsky-2-2-decoder",     # Kandinsky 2.2 (KandinskyV22Pipeline)
    # DeepFloyd IF Series
    "DeepFloyd/IF-I-XL-v1.0",                        # DeepFloyd IF Stage 1 (IFPipeline)
    "DeepFloyd/IF-II-L-v1.0",                        # DeepFloyd IF Stage 2
    "DeepFloyd/IF-III-v1.0",                         # DeepFloyd IF Stage 3
    # Other Mainstream Pipelines
    "playgroundai/playground-v2-1024px-aesthetic",   # Playground V2
    "Lykon/dreamshaper-8",                           # DreamShaper 8
    "prompthero/openjourney-v4",                     # OpenJourney V4
    "SG161222/Realistic_Vision_V6.0_B1_noVAE",       # Realistic Vision V6
    # Other Representative Pipelines
    "CompVis/ldm-text2im-large-256",                 # Latent Diffusion (LDMPipeline)
    "PixArt-alpha/PixArt-XL-2-1024-MS",              # PixArt-Alpha (PixArtAlphaPipeline)
    "stabilityai/stable-diffusion-2-1-unclip",       # Stable unCLIP (StableUnCLIPPipeline)
    "TencentARC/InstantID",                          # IP-Adapter/InstantID (StableDiffusionXLInstantIDPipeline)
    "thu-ml/unidiffuser",                            # UniDiffuser (UniDiffuserPipeline)
    "openai/consistency-decoder",                    # Consistency Models (ConsistencyDecoderPipeline)
    "google/ddpm-cifar10-32",                        # DDPM (DDPMPipeline)
    "stabilityai/stable-diffusion-x4-upscaler",      # Stable Diffusion Upscaler (StableDiffusionLatentUpscalePipeline)
    "Wuerstchen/Wuerstchen",                         # Wuerstchen (WuerstchenPipeline)
    "kolors/kolors",                                 # Kolors (KolorsPipeline)
    "lumina-t2x/lumina-t2x",                         # Lumina-T2X (LuminaT2XPipeline)
    "pag/pag",                                       # PAG (PAGPipeline)
    "a-mused/a-mused",                               # aMUSEd (AMUSEDPipeline)
    "attend-and-excite/attend-and-excite",           # Attend-and-Excite (AttendAndExcitePipeline)
    "self-attention-guidance/self-attention-guidance",# Self-Attention Guidance
    "semantic-guidance/semantic-guidance",           # Semantic Guidance
    "flux/flux",                                     # Flux (FluxPipeline)
    "hunyuan-dit/hunyuan-dit",                       # Hunyuan-DiT (HunyuanDiTPipeline)
    "controlnet/controlnet",                         # ControlNet (ControlNetPipeline)
    "controlnet-xs/controlnet-xs",                   # ControlNet-XS (ControlNetXSPipeline)
    "controlnet-flux/controlnet-flux",               # ControlNet with Flux.1
    "controlnet-hunyuan-dit/controlnet-hunyuan-dit", # ControlNet with Hunyuan-DiT
    "controlnet-sd3/controlnet-sd3",                 # ControlNet with Stable Diffusion 3
    "controlnet-sdxl/controlnet-sdxl",               # ControlNet with Stable Diffusion XL
    "latent-consistency-models/latent-consistency-models", # Latent Consistency Models
    "multi-diffusion/multi-diffusion",               # MultiDiffusion
    "stable-cascade/stable-cascade",                 # Stable Cascade
    # For more models, see: https://huggingface.co/models?pipeline_tag=text-to-image
    "See more models: https://huggingface.co/models?pipeline_tag=text-to-image",
]

# Image-to-Image
supported_image2image_models = [
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "runwayml/stable-diffusion-v1-5",                # StableDiffusionImg2ImgPipeline
    "stabilityai/stable-diffusion-xl-base-1.0",      # StableDiffusionXLImg2ImgPipeline
    "kandinsky-community/kandinsky-2-2-decoder",     # KandinskyV22Img2ImgPipeline
    "DeepFloyd/IF-I-XL-v1.0",                        # IFPipeline
    "stabilityai/stable-diffusion-2-1-unclip",       # StableUnCLIPImg2ImgPipeline
    "TencentARC/InstantID",                          # StableDiffusionXLInstantIDPipeline
    "thu-ml/unidiffuser",                            # UniDiffuserPipeline
    "PixArt-alpha/PixArt-XL-2-1024-MS",              # PixArtAlphaImg2ImgPipeline
    "kolors/kolors",                                 # KolorsPipeline
    "VisualCloze/VisualCloze",                       # VisualClozePipeline
    # For more models, see: https://huggingface.co/models?pipeline_tag=image-to-image
    "See more models: https://huggingface.co/models?pipeline_tag=image-to-image",
]

# Inpainting/Restoration
supported_inpainting_models = [
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "runwayml/stable-diffusion-inpainting",          # StableDiffusionInpaintPipeline
    "stabilityai/stable-diffusion-xl-base-1.0",      # StableDiffusionXLInpaintPipeline
    "DeepFloyd/IF-I-XL-v1.0",                        # IFPipeline
    "diffusers/diffedit-model",                      # DiffEditPipeline
    "timbrooks/instruct-pix2pix",                    # InstructPix2PixPipeline
    "stabilityai/paint-by-example",                  # PaintByExamplePipeline
    "VisualCloze/VisualCloze",                       # VisualClozePipeline
    "le-dits/le-dits",                               # LEDITS++ (LEDITSPlusPlusPipeline)
    # For more models, see: https://huggingface.co/models?pipeline_tag=inpainting
    "See more models: https://huggingface.co/models?pipeline_tag=inpainting",
]

# Super-Resolution
supported_superresolution_models = [
    "stabilityai/stable-diffusion-x4-upscaler",      # StableDiffusionLatentUpscalePipeline
    "DeepFloyd/IF-II-L-v1.0",                        # IFPipeline
    "CompVis/ldm-super-resolution-4x-openimages",    # LDMSuperResolutionPipeline
    "stabilityai/stable-diffusion-2-1-unclip",       # StableUnCLIPImageVariationPipeline
    # For more models, see: https://huggingface.co/models?pipeline_tag=super-resolution
    "See more models: https://huggingface.co/models?pipeline_tag=super-resolution",
]

# Depth2Image / Depth Estimation
supported_depth2image_models = [
    "stabilityai/stable-diffusion-2-depth",          # StableDiffusionDepth2ImgPipeline
    "lllyasviel/ControlNet",                         # ControlNetPipeline (depth)
    "TencentARC/InstantID",                          # StableDiffusionXLInstantIDPipeline
    "TencentARC/Marigold",                           # MarigoldPipeline (depth-estimation)
    "VisualCloze/VisualCloze",                       # VisualClozePipeline ([depth,normal,edge,pose]2image)
    # For more models, see: https://huggingface.co/models?pipeline_tag=depth-to-image
    "See more models: https://huggingface.co/models?pipeline_tag=depth-to-image",
]

# Style Transfer / Relighting
supported_style_transfer_models = [
    "TencentARC/VisualCloze",                        # VisualClozePipeline (style transfer, relighting)
    # For more models, see: https://huggingface.co/models?pipeline_tag=style-transfer
    "See more models: https://huggingface.co/models?pipeline_tag=style-transfer",
]

# Image Variation
supported_image_variation_models = [
    "stabilityai/stable-diffusion-2-1-unclip",       # StableUnCLIPImageVariationPipeline
    "stabilityai/stable-diffusion-v1-5",             # StableDiffusionImageVariationPipeline
    "TencentARC/InstantID",                          # StableDiffusionXLInstantIDPipeline
    "thu-ml/unidiffuser",                            # UniDiffuserPipeline
    "openai/unclip",                                 # unCLIPPipeline
    # For more models, see: https://huggingface.co/models?pipeline_tag=image-variation
    "See more models: https://huggingface.co/models?pipeline_tag=image-variation",
]

# 🎥 Video Generation and Transformation

# Text-to-Video
supported_text2video_models = [
    "cerspense/zeroscope-v2-xl",                     # ZeroScope (Text2VideoZeroPipeline)
    "ali-vilab/text-to-video-ms-1.7b",               # Text2Video (TextToVideoPipeline)
    "camenduru/AnimateDiff",                         # AnimateDiffPipeline
    "cogvlab/cogvideoX",                             # CogVideoXPipeline
    "text2video/text2video",                         # Text2VideoPipeline
    # For more models, see: https://huggingface.co/models?pipeline_tag=text-to-video
    "See more models: https://huggingface.co/models?pipeline_tag=text-to-video",
]

# Image-to-Video
supported_image2video_models = [
    "ali-vilab/i2vgen-xl",                           # I2VGen-XL (I2VGenXLPipeline)
    "camenduru/PIA",                                 # PIA (PIAPipeline)
    # For more models, see: https://huggingface.co/models?pipeline_tag=image-to-video
    "See more models: https://huggingface.co/models?pipeline_tag=image-to-video",
]

# Video-to-Video
supported_video2video_models = [
    "ali-vilab/text-to-video-ms-1.7b",               # Text2VideoPipeline (video2video)
    # For more models, see: https://huggingface.co/models?pipeline_tag=video-to-video
    "See more models: https://huggingface.co/models?pipeline_tag=video-to-video",
]

# 🔊 Audio Generation and Processing

# Text-to-Audio
supported_text2audio_models = [
    "cvssp/audioldm",                                # AudioLDM (AudioLDMPipeline)
    "cvssp/audioldm2",                               # AudioLDM2 (AudioLDM2Pipeline)
    "facebook/musicgen",                             # MusicLDM (MusicLDMPipeline)
    "stabilityai/stable-audio",                      # Stable Audio (StableAudioPipeline)
    "thu-ml/unidiffuser",                            # UniDiffuserPipeline
    # For more models, see: https://huggingface.co/models?pipeline_tag=text-to-audio
    "See more models: https://huggingface.co/models?pipeline_tag=text-to-audio",
]

# Unconditional Audio Generation
supported_unconditional_audio_models = [
    "harmonai/dance-diffusion",                      # Dance Diffusion (DanceDiffusionPipeline)
    "thu-ml/unidiffuser",                            # UniDiffuserPipeline
    # For more models, see: https://huggingface.co/models?pipeline_tag=unconditional-audio-generation
    "See more models: https://huggingface.co/models?pipeline_tag=unconditional-audio-generation",
]

# 🧠 Unconditional Generation

# Image
supported_unconditional_image_models = [
    "openai/consistency-decoder",                    # Consistency Models (ConsistencyDecoderPipeline)
    "google/ddpm-cifar10-32",                        # DDPM (DDPMPipeline)
    "thu-ml/unidiffuser",                            # UniDiffuserPipeline
    "latent-diffusion/latent-diffusion",             # LatentDiffusionPipeline
    "facebook/ddim",                                 # DDIMPipeline
    # For more models, see: https://huggingface.co/models?pipeline_tag=unconditional-image-generation
    "See more models: https://huggingface.co/models?pipeline_tag=unconditional-image-generation",
]

# Audio
# Already merged into supported_unconditional_audio_models

# 🧩 Other Task Scenarios

# Image-to-Text
supported_image2text_models = [
    "thu-ml/unidiffuser",                            # UniDiffuserPipeline
    # For more models, see: https://huggingface.co/models?pipeline_tag=image-to-text
    "See more models: https://huggingface.co/models?pipeline_tag=image-to-text",
]

# Text Variation
supported_text_variation_models = [
    "thu-ml/unidiffuser",                            # UniDiffuserPipeline
    # For more models, see: https://huggingface.co/models?pipeline_tag=text-variation
    "See more models: https://huggingface.co/models?pipeline_tag=text-variation",
]

# 3D Generation (Text-to-3D / Image-to-3D)
supported_text_to_3d_models = [
    "openai/shap-e",                                 # Shap-E (ShapEPipeline)
    # For more models, see: https://huggingface.co/models?pipeline_tag=text-to-3d
    "See more models: https://huggingface.co/models?pipeline_tag=text-to-3d",
]
supported_image_to_3d_models = [
    "openai/shap-e",                                 # Shap-E (ShapEPipeline)
    # For more models, see: https://huggingface.co/models?pipeline_tag=image-to-3d
    "See more models: https://huggingface.co/models?pipeline_tag=image-to-3d",
]

# Value-Guided Planning
supported_value_guided_planning_models = [
    "thu-ml/value-guided-planning",                  # Value-guided planning (ValueGuidedPlanningPipeline)
    # For more models, see: https://huggingface.co/models?pipeline_tag=value-guided-planning
    "See more models: https://huggingface.co/models?pipeline_tag=value-guided-planning",
]

# Visual Cloze and Virtual Try-on
supported_visualcloze_models = [
    "TencentARC/VisualCloze",                        # VisualClozePipeline
    # For more models, see: https://huggingface.co/models?pipeline_tag=visual-cloze
    "See more models: https://huggingface.co/models?pipeline_tag=visual-cloze",
]

# Miscellaneous: Normal Estimation, Pose Estimation, Edge Detection, etc.
supported_misc_models = [
    "TencentARC/Marigold",                           # MarigoldPipeline (normals-estimation, depth-estimation, intrinsic-decomposition)
    "TencentARC/VisualCloze",                        # VisualClozePipeline ([depth,normal,edge,pose]2image, virtual try-on, etc.)
    # For more models, see: https://huggingface.co/models
    "See more models: https://huggingface.co/models",
]
