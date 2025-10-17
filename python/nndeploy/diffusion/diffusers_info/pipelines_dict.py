#!/usr/bin/env python3
"""
Diffusers Pipeline Dictionary
"""

DIFFUSERS_PIPELINES_DICT = {
    "AllegroPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['rhymes-ai/Allegro'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "AltDiffusionImg2ImgPipeline": {
        "function": "image2image",
        "pretrained_model_paths": ['BAAI/AltDiffusion-m9'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "AltDiffusionPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['BAAI/AltDiffusion-m9'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "AmusedImg2ImgPipeline": {
        "function": "image2image",
        "pretrained_model_paths": ['amused/amused-512'],
        "supports_safetensors": True,
        "model_size_category": "small (<2GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "AmusedInpaintPipeline": {
        "function": "inpainting",
        "pretrained_model_paths": ['amused/amused-512'],
        "supports_safetensors": True,
        "model_size_category": "small (<2GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "AmusedPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['amused/amused-512'],
        "supports_safetensors": True,
        "model_size_category": "small (<2GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "AnimateDiffControlNetPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['wangfuyun/AnimateLCM', 'SG161222/Realistic_Vision_V5.1_noVAE', 'stabilityai/sd-vae-ft-mse', 'lllyasviel/Annotators'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "AnimateDiffPAGPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['SG161222/Realistic_Vision_V5.1_noVAE'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "AnimateDiffPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['guoyww/animatediff-motion-adapter-v1-5-2', 'frankjoshua/toonyou_beta6'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "AnimateDiffSDXLPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['stabilityai/stable-diffusion-xl-base-1.0', 'a-r-r-o-w/animatediff-motion-adapter-sdxl-beta'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "AnimateDiffSparseControlNetPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['SG161222/Realistic_Vision_V5.1_noVAE'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "AnimateDiffVideoToVideoControlNetPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['SG161222/Realistic_Vision_V5.1_noVAE', 'stabilityai/sd-vae-ft-mse', 'wangfuyun/AnimateLCM', 'lllyasviel/sd-controlnet-openpose', 'lllyasviel/Annotators'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "AnimateDiffVideoToVideoPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['guoyww/animatediff-motion-adapter-v1-5-2', 'SG161222/Realistic_Vision_V5.1_noVAE'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "AudioLDM2Pipeline": {
        "function": "text2audio",
        "pretrained_model_paths": ['anhnct/audioldm2_gigaspeech', 'cvssp/audioldm2'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "AudioLDMPipeline": {
        "function": "text2audio",
        "pretrained_model_paths": ['cvssp/audioldm-s-full-v2'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "AuraFlowPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['fal/AuraFlow'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "AutoPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['stabilityai/stable-diffusion-3-medium-diffusers'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "BlipDiffusionControlNetPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['Salesforce/blipdiffusion-controlnet'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "BlipDiffusionPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['Salesforce/blipdiffusion'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "BriaPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['briaai/BRIA-3.2'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "ChromaImg2ImgPipeline": {
        "function": "image2image",
        "pretrained_model_paths": ['lodestones/Chroma'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "ChromaPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['lodestones/Chroma'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "CogVideoXFunControlPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['alibaba-pai/CogVideoX-Fun-V1.1-5b-Pose'],
        "supports_safetensors": True,
        "model_size_category": "large (5-10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "CogVideoXImageToVideoPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['THUDM/CogVideoX-5b-I2V'],
        "supports_safetensors": True,
        "model_size_category": "large (5-10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "CogVideoXPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['THUDM/CogVideoX-2b'],
        "supports_safetensors": True,
        "model_size_category": "small (<2GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "CogVideoXVideoToVideoPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['THUDM/CogVideoX-5b'],
        "supports_safetensors": True,
        "model_size_category": "large (5-10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "CogView3PlusPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['THUDM/CogView3-Plus-3B'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "CogView4ControlPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['THUDM/CogView4-6B-Control'],
        "supports_safetensors": True,
        "model_size_category": "large (5-10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "CogView4Pipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['THUDM/CogView4-6B'],
        "supports_safetensors": True,
        "model_size_category": "large (5-10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "ConsisIDPipeline": {
        "function": "image2video",
        "pretrained_model_paths": ['BestWishYsh/ConsisID-preview'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "ConsistencyModelPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['openai/diffusers-cd_imagenet64_l2'],
        "supports_safetensors": True,
        "model_size_category": "small (<2GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "Cosmos2TextToImagePipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['nvidia/Cosmos-Predict2-2B-Text2Image'],
        "supports_safetensors": True,
        "model_size_category": "small (<2GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "Cosmos2VideoToWorldPipeline": {
        "function": "image2video",
        "pretrained_model_paths": ['nvidia/Cosmos-Predict2-2B-Video2World'],
        "supports_safetensors": True,
        "model_size_category": "small (<2GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "CosmosTextToWorldPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['nvidia/Cosmos-1.0-Diffusion-7B-Text2World'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "CosmosVideoToWorldPipeline": {
        "function": "video2video",
        "pretrained_model_paths": ['nvidia/Cosmos-1.0-Diffusion-7B-Video2World'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "DiffusionPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['cerspense/zeroscope_v2_576w', 'cerspense/zeroscope_v2_XL'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "EasyAnimateControlPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['alibaba-pai/EasyAnimateV5.1-12b-zh-Control-diffusers'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "EasyAnimateInpaintPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['alibaba-pai/EasyAnimateV5.1-12b-zh-InP-diffusers'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "EasyAnimatePipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['alibaba-pai/EasyAnimateV5.1-7b-zh-diffusers'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "FlaxStableDiffusionControlNetPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['stable-diffusion-v1-5/stable-diffusion-v1-5', 'lllyasviel/sd-controlnet-canny'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "JAX",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "FlaxStableDiffusionImg2ImgPipeline": {
        "function": "image2image",
        "pretrained_model_paths": ['CompVis/stable-diffusion-v1-4'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "JAX",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "FlaxStableDiffusionInpaintPipeline": {
        "function": "inpainting",
        "pretrained_model_paths": ['xvjiarui/stable-diffusion-2-inpainting'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "JAX",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "FlaxStableDiffusionPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['stable-diffusion-v1-5/stable-diffusion-v1-5'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "JAX",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "FluxControlImg2ImgPipeline": {
        "function": "image2image",
        "pretrained_model_paths": ['black-forest-labs/FLUX.1-Canny-dev'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForImage2Image']
    },
    "FluxControlInpaintPipeline": {
        "function": "inpainting",
        "pretrained_model_paths": ['LiheYoung/depth-anything-large-hf', 'black-forest-labs/FLUX.1-Depth-dev', 'sayakpaul/FLUX.1-Depth-dev-nf4'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForInpainting']
    },
    "FluxControlNetImg2ImgPipeline": {
        "function": "image2image",
        "pretrained_model_paths": ['black-forest-labs/FLUX.1-schnell', 'InstantX/FLUX.1-dev-Controlnet-Canny-alpha'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForImage2Image']
    },
    "FluxControlNetInpaintPipeline": {
        "function": "inpainting",
        "pretrained_model_paths": ['black-forest-labs/FLUX.1-schnell', 'InstantX/FLUX.1-dev-controlnet-canny'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForInpainting']
    },
    "FluxControlNetPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['example/fluxcontrolnetpipeline'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "FluxControlPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['black-forest-labs/FLUX.1-Canny-dev'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "FluxFillPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['black-forest-labs/FLUX.1-Fill-dev'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "FluxImg2ImgPipeline": {
        "function": "image2image",
        "pretrained_model_paths": ['black-forest-labs/FLUX.1-schnell'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForImage2Image']
    },
    "FluxInpaintPipeline": {
        "function": "inpainting",
        "pretrained_model_paths": ['black-forest-labs/FLUX.1-schnell'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForInpainting']
    },
    "FluxKontextInpaintPipeline": {
        "function": "inpainting",
        "pretrained_model_paths": ['black-forest-labs/FLUX.1-Kontext-dev'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "FluxKontextPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['black-forest-labs/FLUX.1-Kontext-dev'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "FluxPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['black-forest-labs/FLUX.1-schnell'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "FluxPriorReduxPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['example/fluxpriorreduxpipeline'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "HiDreamImagePipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['HiDream-ai/HiDream-I1-Full', 'meta-llama/Meta-Llama-3.1-8B-Instruct'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "HunyuanDiTControlNetPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Canny', 'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "HunyuanDiTPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['Tencent-Hunyuan/HunyuanDiT-Diffusers'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "HunyuanSkyreelsImageToVideoPipeline": {
        "function": "image2video",
        "pretrained_model_paths": ['Skywork/SkyReels-V1-Hunyuan-I2V', 'hunyuanvideo-community/HunyuanVideo'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "HunyuanVideoFramepackPipeline": {
        "function": "image2video",
        "pretrained_model_paths": ['lllyasviel/flux_redux_bfl', 'hunyuanvideo-community/HunyuanVideo', 'lllyasviel/FramePackI2V_HY'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "HunyuanVideoImageToVideoPipeline": {
        "function": "image2video",
        "pretrained_model_paths": ['hunyuanvideo-community/HunyuanVideo-I2V'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "HunyuanVideoPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['hunyuanvideo-community/HunyuanVideo'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "I2VGenXLPipeline": {
        "function": "image2video",
        "pretrained_model_paths": ['ali-vilab/i2vgen-xl'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "IFImg2ImgPipeline": {
        "function": "image2image",
        "pretrained_model_paths": ['DeepFloyd/IF-I-XL-v1.0', 'DeepFloyd/IF-II-L-v1.0'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForImage2Image']
    },
    "IFInpaintingPipeline": {
        "function": "inpainting",
        "pretrained_model_paths": ['DeepFloyd/IF-I-XL-v1.0', 'DeepFloyd/IF-II-L-v1.0'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForInpainting']
    },
    "IFPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['DeepFloyd/IF-I-XL-v1.0', 'DeepFloyd/IF-II-L-v1.0'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "KandinskyImg2ImgPipeline": {
        "function": "image2image",
        "pretrained_model_paths": ['kandinsky-community/kandinsky-2-1-prior', 'kandinsky-community/kandinsky-2-1'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForImage2Image']
    },
    "KandinskyInpaintPipeline": {
        "function": "inpainting",
        "pretrained_model_paths": ['kandinsky-community/kandinsky-2-1-inpaint', 'kandinsky-community/kandinsky-2-1-prior'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForInpainting']
    },
    "KandinskyPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['kandinsky-community/kandinsky-2-1-prior', 'kandinsky-community/kandinsky-2-1'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "KandinskyV22Img2ImgPipeline": {
        "function": "image2image",
        "pretrained_model_paths": ['kandinsky-community/kandinsky-2-2-prior', 'kandinsky-community/kandinsky-2-2-decoder'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForImage2Image']
    },
    "KandinskyV22InpaintPipeline": {
        "function": "inpainting",
        "pretrained_model_paths": ['kandinsky-community/kandinsky-2-2-prior', 'kandinsky-community/kandinsky-2-2-decoder-inpaint'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForInpainting']
    },
    "KandinskyV22Pipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['kandinsky-community/kandinsky-2-2-decoder, torch_dtype=torch.float16', 'kandinsky-community/kandinsky-2-2-prior'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "KandinskyV22PriorEmb2EmbPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['kandinsky-community/kandinsky-2-2-prior', 'kandinsky-community/kandinsky-2-2-controlnet-depth'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "KandinskyV22PriorPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['kandinsky-community/kandinsky-2-2-prior', 'kandinsky-community/kandinsky-2-2-controlnet-depth'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "KolorsImg2ImgPipeline": {
        "function": "image2image",
        "pretrained_model_paths": ['Kwai-Kolors/Kolors-diffusers'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForImage2Image']
    },
    "KolorsPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['Kwai-Kolors/Kolors-diffusers'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "LEditsPPPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['stabilityai/stable-diffusion-xl-base-1.0'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "LTXConditionPipeline": {
        "function": "video2video",
        "pretrained_model_paths": ['Lightricks/LTX-Video-0.9.5'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "LTXImageToVideoPipeline": {
        "function": "image2video",
        "pretrained_model_paths": ['Lightricks/LTX-Video'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "LTXPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['Lightricks/LTX-Video'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "LattePipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['maxin-cn/Latte-1'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "Lumina2Pipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['Alpha-VLLM/Lumina-Image-2.0'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "LuminaPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['Alpha-VLLM/Lumina-Next-SFT-diffusers'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "MarigoldDepthPipeline": {
        "function": "depth_estimation",
        "pretrained_model_paths": ['prs-eth/marigold-depth-v1-1'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "MarigoldIntrinsicsPipeline": {
        "function": "intrinsic_decomposition",
        "pretrained_model_paths": ['prs-eth/marigold-iid-lighting-v1-1', 'prs-eth/marigold-iid-appearance-v1-1'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "MarigoldNormalsPipeline": {
        "function": "normal_estimation",
        "pretrained_model_paths": ['prs-eth/marigold-normals-v1-1'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "MochiPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['genmo/mochi-1-preview'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "MusicLDMPipeline": {
        "function": "text2audio",
        "pretrained_model_paths": ['ucsd-reach/musicldm'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "OmniGenPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['Shitao/OmniGen-v1-diffusers'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "PIAPipeline": {
        "function": "image2video",
        "pretrained_model_paths": ['openmmlab/PIA-condition-adapter', 'SG161222/Realistic_Vision_V6.0_B1_noVAE'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "PixArtAlphaPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['PixArt-alpha/PixArt-XL-2-1024-MS'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "PixArtSigmaPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['PixArt-alpha/PixArt-Sigma-XL-2-1024-MS'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "QwenImageControlNetPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['Qwen/Qwen-Image', 'InstantX/Qwen-Image-ControlNet-Union'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "QwenImageEditPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['Qwen/Qwen-Image-Edit'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "QwenImageImg2ImgPipeline": {
        "function": "image2image",
        "pretrained_model_paths": ['Qwen/Qwen-Image'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "QwenImageInpaintPipeline": {
        "function": "inpainting",
        "pretrained_model_paths": ['Qwen/Qwen-Image'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "QwenImagePipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['Qwen/Qwen-Image'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "SanaControlNetPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['ishan24/Sana_600M_1024px_ControlNetPlus_diffusers'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "SanaPAGPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "SanaPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "SanaSprintImg2ImgPipeline": {
        "function": "image2image",
        "pretrained_model_paths": ['Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "SanaSprintPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "SkyReelsV2DiffusionForcingImageToVideoPipeline": {
        "function": "image2video",
        "pretrained_model_paths": ['Skywork/SkyReels-V2-DF-14B-720P-Diffusers'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "SkyReelsV2DiffusionForcingPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['Skywork/SkyReels-V2-DF-14B-720P-Diffusers'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "SkyReelsV2DiffusionForcingVideoToVideoPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['Skywork/SkyReels-V2-DF-14B-720P-Diffusers'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "SkyReelsV2ImageToVideoPipeline": {
        "function": "image2video",
        "pretrained_model_paths": ['Skywork/SkyReels-V2-I2V-14B-720P-Diffusers'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "SkyReelsV2Pipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['Skywork/SkyReels-V2-T2V-14B-720P-Diffusers'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "StableAudioPipeline": {
        "function": "text2audio",
        "pretrained_model_paths": ['stabilityai/stable-audio-open-1.0'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "StableCascadeCombinedPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['stabilityai/stable-cascade'],
        "supports_safetensors": True,
        "model_size_category": "large (5-10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "StableCascadePriorPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['stabilityai/stable-cascade-prior'],
        "supports_safetensors": True,
        "model_size_category": "large (5-10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "StableDiffusion3ControlNetInpaintingPipeline": {
        "function": "inpainting",
        "pretrained_model_paths": ['alimama-creative/SD3-Controlnet-Inpainting', 'stabilityai/stable-diffusion-3-medium-diffusers'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForInpainting']
    },
    "StableDiffusion3ControlNetPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['InstantX/SD3-Controlnet-Canny', 'stabilityai/stable-diffusion-3-medium-diffusers'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "StableDiffusion3InpaintPipeline": {
        "function": "inpainting",
        "pretrained_model_paths": ['stabilityai/stable-diffusion-3-medium-diffusers'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForInpainting']
    },
    "StableDiffusion3PAGImg2ImgPipeline": {
        "function": "image2image",
        "pretrained_model_paths": ['stabilityai/stable-diffusion-3-medium-diffusers'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForImage2Image']
    },
    "StableDiffusion3Pipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['stabilityai/stable-diffusion-3-medium-diffusers'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "StableDiffusionAdapterPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['TencentARC/t2iadapter_color_sd14v1', 'CompVis/stable-diffusion-v1-4'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "StableDiffusionAttendAndExcitePipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['CompVis/stable-diffusion-v1-4'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "StableDiffusionControlNetImg2ImgPipeline": {
        "function": "image2image",
        "pretrained_model_paths": ['stable-diffusion-v1-5/stable-diffusion-v1-5', 'lllyasviel/sd-controlnet-canny'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForImage2Image']
    },
    "StableDiffusionControlNetInpaintPipeline": {
        "function": "inpainting",
        "pretrained_model_paths": ['stable-diffusion-v1-5/stable-diffusion-v1-5', 'lllyasviel/control_v11p_sd15_inpaint'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForInpainting']
    },
    "StableDiffusionControlNetPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['stable-diffusion-v1-5/stable-diffusion-v1-5', 'lllyasviel/sd-controlnet-canny'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "StableDiffusionControlNetXSPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['UmerHA/Testing-ConrolNetXS-SD2.1-canny', 'stabilityai/stable-diffusion-2-1-base'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "StableDiffusionDiffEditPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['stabilityai/stable-diffusion-2-1'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "StableDiffusionGLIGENPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['masterful/gligen-1-4-inpainting-text-box', 'masterful/gligen-1-4-generation-text-box'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "StableDiffusionGLIGENTextImagePipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['anhnct/Gligen_Inpainting_Text_Image', 'anhnct/Gligen_Text_Image'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "StableDiffusionImg2ImgPipeline": {
        "function": "image2image",
        "pretrained_model_paths": ['stable-diffusion-v1-5/stable-diffusion-v1-5'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForImage2Image']
    },
    "StableDiffusionLDM3DPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['Intel/ldm3d-4c'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "StableDiffusionPanoramaPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['example/stablediffusionpanoramapipeline'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "StableDiffusionParadigmsPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['runwayml/stable-diffusion-v1-5'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "StableDiffusionPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['stable-diffusion-v1-5/stable-diffusion-v1-5'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "StableDiffusionPix2PixZeroPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['example/stablediffusionpix2pixzeropipeline'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "StableDiffusionSAGPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['stable-diffusion-v1-5/stable-diffusion-v1-5'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "StableDiffusionXLAdapterPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['stabilityai/stable-diffusion-xl-base-1.0', 'Adapter/t2iadapter'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "StableDiffusionXLControlNetImg2ImgPipeline": {
        "function": "depth_estimation",
        "pretrained_model_paths": ['stabilityai/stable-diffusion-xl-base-1.0', 'madebyollin/sdxl-vae-fp16-fix', 'Intel/dpt-hybrid-midas', 'diffusers/controlnet-depth-sdxl-1.0-small'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForImage2Image']
    },
    "StableDiffusionXLControlNetInpaintPipeline": {
        "function": "inpainting",
        "pretrained_model_paths": ['diffusers/controlnet-canny-sdxl-1.0', 'stabilityai/stable-diffusion-xl-base-1.0'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForInpainting']
    },
    "StableDiffusionXLControlNetPAGImg2ImgPipeline": {
        "function": "depth_estimation",
        "pretrained_model_paths": ['stabilityai/stable-diffusion-xl-base-1.0', 'madebyollin/sdxl-vae-fp16-fix', 'Intel/dpt-hybrid-midas', 'diffusers/controlnet-depth-sdxl-1.0-small'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForImage2Image']
    },
    "StableDiffusionXLControlNetPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['diffusers/controlnet-canny-sdxl-1.0', 'stabilityai/stable-diffusion-xl-base-1.0', 'madebyollin/sdxl-vae-fp16-fix'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "StableDiffusionXLControlNetUnionImg2ImgPipeline": {
        "function": "image2image",
        "pretrained_model_paths": ['brad-twinkl/controlnet-union-sdxl-1.0-promax', 'stabilityai/stable-diffusion-xl-base-1.0', 'madebyollin/sdxl-vae-fp16-fix'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForImage2Image']
    },
    "StableDiffusionXLControlNetUnionInpaintPipeline": {
        "function": "inpainting",
        "pretrained_model_paths": ['brad-twinkl/controlnet-union-sdxl-1.0-promax', 'stabilityai/stable-diffusion-xl-base-1.0', 'madebyollin/sdxl-vae-fp16-fix'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForInpainting']
    },
    "StableDiffusionXLControlNetUnionPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['xinsir/controlnet-union-sdxl-1.0', 'stabilityai/stable-diffusion-xl-base-1.0', 'madebyollin/sdxl-vae-fp16-fix', 'lllyasviel/Annotators'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "StableDiffusionXLControlNetXSPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['UmerHA/Testing-ConrolNetXS-SDXL-canny', 'stabilityai/stable-diffusion-xl-base-1.0', 'madebyollin/sdxl-vae-fp16-fix'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "StableDiffusionXLImg2ImgPipeline": {
        "function": "image2image",
        "pretrained_model_paths": ['stabilityai/stable-diffusion-xl-refiner-1.0'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForImage2Image']
    },
    "StableDiffusionXLInpaintPipeline": {
        "function": "inpainting",
        "pretrained_model_paths": ['stabilityai/stable-diffusion-xl-base-1.0'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForInpainting']
    },
    "StableDiffusionXLInstructPix2PixPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['diffusers/sdxl-instructpix2pix-768'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "StableDiffusionXLKDiffusionPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['stabilityai/stable-diffusion-xl-base-1.0'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "StableDiffusionXLPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['stabilityai/stable-diffusion-xl-base-1.0'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "StableUnCLIPImg2ImgPipeline": {
        "function": "image2image",
        "pretrained_model_paths": ['stabilityai/stable-diffusion-2-1-unclip-small'],
        "supports_safetensors": True,
        "model_size_category": "small (<2GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "StableUnCLIPPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['fusing/stable-unclip-2-1-l'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "StableVideoDiffusionPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['stabilityai/stable-video-diffusion-img2vid-xt'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "TextToVideoSDPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['damo-vilab/text-to-video-ms-1.7b'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "VisualClozeGenerationPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['VisualCloze/VisualClozePipeline-384'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "VisualClozePipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['VisualCloze/VisualClozePipeline-384'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "WanImageToVideoPipeline": {
        "function": "image2video",
        "pretrained_model_paths": ['Wan-AI/Wan2.1-I2V-14B-480P-Diffusers'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "WanPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['Wan-AI/Wan2.1-T2V-14B-Diffusers'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "WanVACEPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['Wan-AI/Wan2.1-VACE-1.3B-diffusers'],
        "supports_safetensors": True,
        "model_size_category": "medium (2-5GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "WanVideoToVideoPipeline": {
        "function": "text2video",
        "pretrained_model_paths": ['Wan-AI/Wan2.1-T2V-1.3B-Diffusers'],
        "supports_safetensors": True,
        "model_size_category": "extra_large (>10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
    "WuerstchenCombinedPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['warp-ai/Wuerstchen'],
        "supports_safetensors": True,
        "model_size_category": "large (5-10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": True,
        "auto_pipeline_types": ['AutoPipelineForText2Image']
    },
    "WuerstchenPriorPipeline": {
        "function": "text2image",
        "pretrained_model_paths": ['warp-ai/wuerstchen-prior'],
        "supports_safetensors": True,
        "model_size_category": "large (5-10GB)",
        "backend": "PyTorch",
        "supports_auto_pipeline": False,
        "auto_pipeline_types": []
    },
}

    
