#!/usr/bin/env python3
"""
Extract pipeline information from diffusers_pipeline_md.py and generate diffusers_pipelines_dict.py
"""

import re
import ast
from pipeline_md import DIFFUSERS_PIPELINE_MD

# Auto pipeline mappings from auto_pipeline.py
AUTO_TEXT2IMAGE_PIPELINES = {
    "StableDiffusionPipeline", "StableDiffusionXLPipeline", "StableDiffusion3Pipeline", 
    "StableDiffusion3PAGPipeline", "IFPipeline", "HunyuanDiTPipeline", "HunyuanDiTPAGPipeline",
    "KandinskyCombinedPipeline", "KandinskyV22CombinedPipeline", "Kandinsky3Pipeline",
    "StableDiffusionControlNetPipeline", "StableDiffusionXLControlNetPipeline", 
    "StableDiffusionXLControlNetUnionPipeline", "StableDiffusion3ControlNetPipeline",
    "WuerstchenCombinedPipeline", "StableCascadeCombinedPipeline", "LatentConsistencyModelPipeline",
    "PixArtAlphaPipeline", "PixArtSigmaPipeline", "SanaPipeline", "SanaPAGPipeline",
    "StableDiffusionPAGPipeline", "StableDiffusionControlNetPAGPipeline", 
    "StableDiffusionXLPAGPipeline", "StableDiffusionXLControlNetPAGPipeline",
    "PixArtSigmaPAGPipeline", "AuraFlowPipeline", "FluxPipeline", "FluxControlPipeline",
    "FluxControlNetPipeline", "FluxKontextPipeline", "LuminaPipeline", "Lumina2Pipeline",
    "ChromaPipeline", "CogView3PlusPipeline", "CogView4Pipeline", "CogView4ControlPipeline",
    "KolorsPipeline", "KolorsPAGPipeline"
}

AUTO_IMAGE2IMAGE_PIPELINES = {
    "StableDiffusionImg2ImgPipeline", "StableDiffusionXLImg2ImgPipeline", 
    "StableDiffusion3Img2ImgPipeline", "StableDiffusion3PAGImg2ImgPipeline",
    "IFImg2ImgPipeline", "KandinskyImg2ImgCombinedPipeline", "KandinskyV22Img2ImgCombinedPipeline",
    "Kandinsky3Img2ImgPipeline", "StableDiffusionControlNetImg2ImgPipeline",
    "StableDiffusionPAGImg2ImgPipeline", "StableDiffusionXLControlNetImg2ImgPipeline",
    "StableDiffusionXLControlNetUnionImg2ImgPipeline", "StableDiffusionXLPAGImg2ImgPipeline",
    "StableDiffusionXLControlNetPAGImg2ImgPipeline", "LatentConsistencyModelImg2ImgPipeline",
    "FluxImg2ImgPipeline", "FluxControlNetImg2ImgPipeline", "FluxControlImg2ImgPipeline",
    "KolorsImg2ImgPipeline"
}

AUTO_INPAINT_PIPELINES = {
    "StableDiffusionInpaintPipeline", "StableDiffusionXLInpaintPipeline",
    "StableDiffusion3InpaintPipeline", "IFInpaintingPipeline", 
    "KandinskyInpaintCombinedPipeline", "KandinskyV22InpaintCombinedPipeline",
    "StableDiffusionControlNetInpaintPipeline", "StableDiffusionControlNetPAGInpaintPipeline",
    "StableDiffusionXLControlNetInpaintPipeline", "StableDiffusionXLControlNetUnionInpaintPipeline",
    "StableDiffusion3ControlNetInpaintingPipeline", "StableDiffusionXLPAGInpaintPipeline",
    "FluxInpaintPipeline", "FluxControlNetInpaintPipeline", "FluxControlInpaintPipeline",
    "StableDiffusionPAGInpaintPipeline"
}

# Decoder pipelines (also support auto pipeline)
AUTO_DECODER_PIPELINES = {
    "KandinskyPipeline", "KandinskyV22Pipeline", "WuerstchenDecoderPipeline", 
    "StableCascadeDecoderPipeline", "KandinskyImg2ImgPipeline", "KandinskyV22Img2ImgPipeline",
    "KandinskyInpaintPipeline", "KandinskyV22InpaintPipeline"
}

def extract_pipeline_class_name(code):
    """Extract pipeline class name from import statements or instantiation"""
    # Look for import statements like: from diffusers import SomePipeline
    import_matches = re.findall(r'from diffusers import[^#\n]*?(\w+Pipeline)', code)
    if import_matches:
        return import_matches[0]
    
    # Look for instantiation like: SomePipeline.from_pretrained
    instantiation_match = re.search(r'(\w+Pipeline)\.from_pretrained', code)
    if instantiation_match:
        return instantiation_match.group(1)
    
    return None

def extract_model_paths(code):
    """Extract model paths from from_pretrained calls"""
    model_paths = []
    
    # Find all from_pretrained calls with string arguments, including multiline
    # Pattern to handle various formats including multiline with ...
    patterns = [
        r'\.from_pretrained\(\s*["\']([^"\']+)["\']',  # Single line
        r'\.from_pretrained\(\s*\n?\s*["\']([^"\']+)["\']',  # Multiline simple
        r'\.from_pretrained\(\s*\n?\s*\.\.\.\s*["\']([^"\']+)["\']',  # With ... continuation
        r'model_id_or_path\s*=\s*["\']([^"\']+)["\']',  # Variable assignment
        r'repo_id\s*=\s*["\']([^"\']+)["\']',  # repo_id assignment
        r'model_id\s*=\s*["\']([^"\']+)["\']',  # model_id assignment
    ]
    
    all_matches = []
    for pattern in patterns:
        matches = re.findall(pattern, code, re.MULTILINE | re.DOTALL)
        all_matches.extend(matches)
    
    for match in all_matches:
        # Skip URLs, local paths, but keep valid model paths
        if (not match.startswith('http') and 
            not match.startswith('./') and 
            not match.startswith('../') and
            not match.lower().startswith('example/') and
            ('/' in match or match.count('-') >= 2)):  # Accept paths with / or models with multiple -
            model_paths.append(match)
    
    # If no real model paths found, create example path based on pipeline class name
    if not model_paths:
        pipeline_class = extract_pipeline_class_name(code)
        if pipeline_class:
            return ['example/' + pipeline_class.lower()]
        else:
            return ['example/unknownpipeline']
    
    return list(set(model_paths))  # Remove duplicates

def determine_auto_pipeline_support(pipeline_name):
    """Determine if pipeline supports auto pipeline and which types"""
    auto_pipeline_types = []
    
    if pipeline_name in AUTO_TEXT2IMAGE_PIPELINES:
        auto_pipeline_types.append('AutoPipelineForText2Image')
    if pipeline_name in AUTO_IMAGE2IMAGE_PIPELINES:
        auto_pipeline_types.append('AutoPipelineForImage2Image')
    if pipeline_name in AUTO_INPAINT_PIPELINES:
        auto_pipeline_types.append('AutoPipelineForInpainting')
    if pipeline_name in AUTO_DECODER_PIPELINES:
        # Decoder pipelines support their respective types
        if 'Img2Img' in pipeline_name:
            auto_pipeline_types.append('AutoPipelineForImage2Image')
        elif 'Inpaint' in pipeline_name:
            auto_pipeline_types.append('AutoPipelineForInpainting')
        else:
            auto_pipeline_types.append('AutoPipelineForText2Image')
    
    supports_auto_pipeline = len(auto_pipeline_types) > 0
    return supports_auto_pipeline, auto_pipeline_types

def determine_model_size_category(pipeline_name, model_paths):
    """Determine model size category based on pipeline name and model paths"""
    pipeline_name_lower = pipeline_name.lower()
    
    # Check model paths for size indicators
    for path in model_paths:
        path_lower = path.lower()
        # Large models
        if any(indicator in path_lower for indicator in ['xl', 'large', '7b', '8b', '12b', '14b', 'xxl']):
            return "extra_large (>10GB)"
        # Medium-large models  
        elif any(indicator in path_lower for indicator in ['5b', '6b', 'sigma', 'base-1.0']):
            return "large (5-10GB)"
        # Small models
        elif any(indicator in path_lower for indicator in ['2b', '1b', '512', 'small', 'tiny']):
            return "small (<2GB)"
    
    # Check pipeline name for size indicators
    if any(indicator in pipeline_name_lower for indicator in ['xl', 'large', '3d', 'video', 'audio']):
        return "extra_large (>10GB)"
    elif any(indicator in pipeline_name_lower for indicator in ['sigma', 'cascade', 'wuerstchen']):
        return "large (5-10GB)"
    elif any(indicator in pipeline_name_lower for indicator in ['amused', 'consistency', 'ddim', 'ddpm']):
        return "small (<2GB)"
    
    # Default to medium
    return "medium (2-5GB)"

def determine_backend(pipeline_name, code):
    """Determine backend based on pipeline name and code content"""
    pipeline_name_lower = pipeline_name.lower()
    
    # Check for JAX/Flax indicators
    if 'flax' in pipeline_name_lower or 'jax' in code or 'flax' in code:
        return "JAX"
    
    # Check for specific backend mentions in code
    if 'torch_dtype' in code or 'torch.float' in code or 'torch.bfloat' in code:
        return "PyTorch"
    
    # Default to PyTorch
    return "PyTorch"

def determine_function_type(code, pipeline_name):
    """Determine the function type based on pipeline name and code content"""
    pipeline_name_lower = pipeline_name.lower()
    
    # Check code content first for more accurate detection
    if 'export_to_video' in code or '.frames[0]' in code:
        if 'video=' in code and ('image=' in code or 'init_image' in code):
            return 'video2video'
        elif 'image=' in code or 'init_image' in code:
            return 'image2video'
        else:
            return 'text2video'
    elif '.audios[0]' in code or 'audio_length' in code or 'audio_end_in_s' in code:
        return 'text2audio'
    elif 'depth_map' in code or 'depth_estimation' in code:
        return 'depth_estimation'
    elif 'normals' in code and 'visualize_normals' in code:
        return 'normal_estimation'
    elif 'intrinsics' in code and 'visualize_intrinsics' in code:
        return 'intrinsic_decomposition'
    
    # Check for specific function types based on pipeline names
    if 'video2video' in pipeline_name_lower:
        return 'video2video'
    elif 'image2video' in pipeline_name_lower or 'img2vid' in pipeline_name_lower or 'i2v' in pipeline_name_lower:
        return 'image2video'
    elif 'text2video' in pipeline_name_lower or 'texttovideo' in pipeline_name_lower or 't2v' in pipeline_name_lower:
        return 'text2video'
    elif 'img2img' in pipeline_name_lower or 'image2image' in pipeline_name_lower:
        return 'image2image'
    elif 'inpaint' in pipeline_name_lower:
        return 'inpainting'
    elif 'text2audio' in pipeline_name_lower:
        return 'text2audio'
    elif 'audio' in pipeline_name_lower and 'diffusion' in pipeline_name_lower:
        return 'text2audio'
    elif 'depth' in pipeline_name_lower:
        return 'depth_estimation'
    elif 'normal' in pipeline_name_lower:
        return 'normal_estimation'
    elif 'super' in pipeline_name_lower and 'resolution' in pipeline_name_lower:
        return 'super_resolution'
    elif 'upscale' in pipeline_name_lower:
        return 'super_resolution'
    elif 'animatediff' in pipeline_name_lower and 'controlnet' in pipeline_name_lower:
        return 'text2video'  # AnimateDiff ControlNet is for video generation
    elif 'controlnet' in pipeline_name_lower:
        return 'text2image'  # Most other ControlNet pipelines are text2image
    
    # Check code content for output types
    if '.images[0]' in code:
        return 'text2image'
    
    # Default fallback
    return 'text2image'

def extract_all_pipeline_info():
    """Extract information for all pipelines in the MD file"""
    pipeline_info = {}
    
    for file_path, example_code in DIFFUSERS_PIPELINE_MD.items():
        # Extract pipeline class name
        pipeline_class = extract_pipeline_class_name(example_code)
        if not pipeline_class:
            continue
            
        # Extract model paths
        model_paths = extract_model_paths(example_code)
        if not model_paths:
            # Try to find any model reference
            model_paths = ['example/' + pipeline_class.lower()]
        
        # Determine function type
        function_type = determine_function_type(example_code, pipeline_class)
        
        # Determine auto pipeline support
        supports_auto_pipeline, auto_pipeline_types = determine_auto_pipeline_support(pipeline_class)
        
        # Determine model size category
        model_size_category = determine_model_size_category(pipeline_class, model_paths)
        
        # Determine backend
        backend = determine_backend(pipeline_class, example_code)
        
        pipeline_info[pipeline_class] = {
            'function': function_type,
            'pretrained_model_paths': model_paths,
            'supports_safetensors': True,
            'model_size_category': model_size_category,
            'backend': backend,
            'supports_auto_pipeline': supports_auto_pipeline,
            'auto_pipeline_types': auto_pipeline_types
        }
    
    return pipeline_info

def generate_pipeline_dict():
    """Generate the complete diffusers_pipelines_dict.py file content"""
    pipeline_info = extract_all_pipeline_info()
    
    lines = [
        '#!/usr/bin/env python3',
        '"""',
        'Diffusers Pipeline Dictionary',
        '"""',
        '',
        'DIFFUSERS_PIPELINES_DICT = {'
    ]
    
    for pipeline_name, info in sorted(pipeline_info.items()):
        lines.extend([
            f'    "{pipeline_name}": {{',
            f'        "function": "{info["function"]}",',
            f'        "pretrained_model_paths": {info["pretrained_model_paths"]},',
            f'        "supports_safetensors": {info["supports_safetensors"]},',
            f'        "model_size_category": "{info["model_size_category"]}",',
            f'        "backend": "{info["backend"]}",',
            f'        "supports_auto_pipeline": {info["supports_auto_pipeline"]},',
            f'        "auto_pipeline_types": {info["auto_pipeline_types"]}',
            '    },'
        ])
    
    lines.extend([
        '}',
        '',
        '    ',
        ''
    ])
    
    return '\n'.join(lines)

if __name__ == "__main__":
    # Generate and write the dictionary file
    content = generate_pipeline_dict()
    with open('pipelines_dict.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Generated pipelines_dict.py successfully!")
    
    # Also print summary for verification
    pipeline_info = extract_all_pipeline_info()
    print(f"\nTotal pipelines: {len(pipeline_info)}")
    print("\nFunction distribution:")
    function_counts = {}
    for info in pipeline_info.values():
        func = info['function']
        function_counts[func] = function_counts.get(func, 0) + 1
    
    for func, count in sorted(function_counts.items()):
        print(f"  {func}: {count}")
        
    # Print first few for verification
    print("\nFirst 5 pipelines:")
    for i, (pipeline_name, info) in enumerate(sorted(pipeline_info.items())):
        if i >= 5:
            break
        print(f"  {pipeline_name}: {info['function']} - {info['pretrained_model_paths'][:2]}")
        if len(info['pretrained_model_paths']) > 2:
            print(f"    ... and {len(info['pretrained_model_paths']) - 2} more models")
