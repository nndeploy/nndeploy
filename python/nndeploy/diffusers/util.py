# help function

import os
import shutil
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional, Union

from diffusers import DiffusionPipeline
from huggingface_hub import list_models, snapshot_download, HfApi
from huggingface_hub.utils import HfHubHTTPError

def get_default_cache_dir() -> str:
    """Get default cache directory"""
    hf_home = os.environ.get('HF_HOME')
    if hf_home:
        return os.path.join(hf_home, 'hub')
    
    hf_cache = os.environ.get('HUGGINGFACE_HUB_CACHE')
    if hf_cache:
        return hf_cache
    
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

def get_supported_models(library: str = "diffusers", limit: int = 500, use_online: bool = False) -> List[str]:
    """
    获取支持的diffusers模型列表
    
    Args:
        library: 库名称，默认为"diffusers"
        limit: 返回模型数量限制
        use_online: 是否使用在线获取，False时返回预定义列表
        
    Returns:
        模型ID列表
    """
    # 预定义的主流diffusers模型列表
    # 基于 Hugging Face Diffusers 官方文档支持的主流模型
    # 参考: https://github.com/huggingface/diffusers
    model_ids = [
        # Stable Diffusion 系列
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1", 
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/sdxl-turbo",
        "stabilityai/sd-turbo",
        "CompVis/stable-diffusion-v1-4",
        "stabilityai/stable-diffusion-2-1-base",
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        
        # ControlNet 系列
        "lllyasviel/sd-controlnet-canny",
        "lllyasviel/sd-controlnet-depth", 
        "lllyasviel/sd-controlnet-openpose",
        "lllyasviel/sd-controlnet-scribble",
        "lllyasviel/sd-controlnet-seg",
        "lllyasviel/sd-controlnet-normal",
        "lllyasviel/sd-controlnet-mlsd",
        "lllyasviel/sd-controlnet-hed",
        
        # Stable Diffusion 3 系列
        "stabilityai/stable-diffusion-3-medium-diffusers",
        "stabilityai/stable-diffusion-3.5-large",
        "stabilityai/stable-diffusion-3.5-medium",
        
        # FLUX 系列
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-schnell",
        
        # Kandinsky 系列
        "kandinsky-community/kandinsky-2-1",
        "kandinsky-community/kandinsky-2-2-decoder",
        "kandinsky-community/kandinsky-3",
        
        # DeepFloyd IF 系列
        "DeepFloyd/IF-I-XL-v1.0",
        "DeepFloyd/IF-II-L-v1.0",
        
        # PixArt 系列
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        
        # HunyuanDiT 系列
        "Tencent-Hunyuan/HunyuanDiT-Diffusers",
        "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers",
        
        # 视频生成模型
        "damo-vilab/text-to-video-ms-1.7b",
        "ali-vilab/modelscope-damo-text-to-video-synthesis",
        "cerspense/zeroscope_v2_576w",
        "cerspense/zeroscope_v2_XL",
        "VideoCrafter/VideoCrafter2",
        "THUDM/CogVideoX-2b",
        "THUDM/CogVideoX-5b",
        "genmo/mochi-1-preview",
        "Lightricks/LTX-Video",
        
        # 音频生成模型
        "facebook/musicgen-small",
        "facebook/musicgen-medium", 
        "facebook/musicgen-large",
        "facebook/audiogen-medium",
        "stabilityai/stable-audio-open-1.0",
        
        # 3D生成模型
        "openai/shap-e",
        "openai/point-e",
        
        # 图像修复和编辑
        "runwayml/stable-diffusion-inpainting",
        "timbrooks/instruct-pix2pix",
        "Fantasy-Studio/Paint-by-Example",
        "lllyasviel/control_v11p_sd15_inpaint",
        
        # 图像超分辨率
        "stabilityai/stable-diffusion-x4-upscaler",
        "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
        
        # 特殊用途模型
        "lambdalabs/sd-image-variations-diffusers",
        "nitrosocke/Arcane-Diffusion",
        "dreamlike-art/dreamlike-diffusion-1.0",
        "prompthero/openjourney",
        "hakurei/waifu-diffusion",
        "naclbit/trinart_stable_diffusion_v2",
        
        # LCM (Latent Consistency Model) 系列
        "SimianLuo/LCM_Dreamshaper_v7",
        "latent-consistency/lcm-lora-sdv1-5",
        "latent-consistency/lcm-lora-sdxl",
        
        # AnimateDiff 系列
        "guoyww/animatediff-motion-adapter-v1-5-2",
        "guoyww/animatediff-motion-adapter-sdxl-beta",
        
        # Wuerstchen 系列
        "warp-ai/wuerstchen-prior",
        "warp-ai/wuerstchen",
        
        # Stable Cascade 系列
        "stabilityai/stable-cascade-prior",
        "stabilityai/stable-cascade",
        
        # I2VGen-XL 系列
        "ali-vilab/i2vgen-xl",
        
        # Lumina 系列
        "Alpha-VLLM/Lumina-T2X",
        
        # AuraFlow 系列
        "fal/AuraFlow",
        
        # Sana 系列
        "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        
        # OmniGen 系列
        "Shitao/OmniGen-v1",
        
        # Allegro 系列
        "rhymes-ai/Allegro",
        
        # Mochi 系列
        "genmo/mochi-1-preview",
        
        # CogView 系列
        "THUDM/CogView3-Plus-3B",
        
        # Qwen 系列
        "Qwen/Qwen2-VL-7B-Instruct",
        
        # ConsisID 系列
        "JackAILab/ConsisID-preview",
        
        # Cosmos 系列
        "nvidia/Cosmos-1.0-Diffusion-7B-Text2World",
        
        # EasyAnimate 系列
        "alibaba-pai/EasyAnimateV5-12b-zh-InP",
        
        # HiDream 系列
        "HiDream/hidream-image-v1.0",
        
        # Latte 系列
        "maxin-cn/Latte-1",
        
        # Marigold 系列
        "prs-eth/marigold-depth-lcm-v1-0",
        
        # PIA 系列
        "PIA-Diffusion/PIA",
        
        # SkyReels 系列
        "SkyReels/SkyReels-v2-diffusion-forcing",
        
        # WAN 系列
        "WAN-AI/WAN-diffusion",
        
        # VisualCloze 系列
        "Visual-Cloze/visual-cloze-generation",
        
        # VQ-Diffusion 系列
        "microsoft/vq-diffusion-ithq",
        
        # UniDiffuser 系列
        "thu-ml/unidiffuser-v1",
        
        # VersatileDiffusion 系列
        "shi-labs/versatile-diffusion",
        
        # Chroma 系列
        "chroma-team/chroma-1.5",
        
        # Bria 系列
        "briaai/BRIA-2.3",
        
        # AltDiffusion 系列
        "BAAI/AltDiffusion-m9",
        
        # Amused 系列
        "amused/amused-512",
        
        # LDM 系列
        "CompVis/ldm-text2im-large-256",
        
        # UnCLIP 系列
        "kakaobrain/karlo-v1-alpha",
        
        # Semantic Stable Diffusion
        "runwayml/stable-diffusion-v1-5",
        
        # LEdits++ 系列
        "editing-images/leditspp",
    ]
    
    if use_online:
        try:
            print(f"正在获取 {library} 支持的模型列表...")
            models = list_models(library=library, limit=limit)
            online_model_ids = [model.modelId for model in models]
            print(f"成功获取 {len(online_model_ids)} 个在线模型")
            return online_model_ids
        except HfHubHTTPError as e:
            print(f"网络请求失败: {e}")
            print("使用预定义模型列表")
            return model_ids[:limit] if limit < len(model_ids) else model_ids
        except Exception as e:
            print(f"获取在线模型列表时发生错误: {e}")
            print("使用预定义模型列表")
            return model_ids[:limit] if limit < len(model_ids) else model_ids
    else:
        return model_ids[:limit] if limit < len(model_ids) else model_ids

def check_network_connectivity(timeout: int = 10) -> Dict[str, bool]:
    """
    Check network connectivity status
    
    Args:
        timeout: Connection timeout
        
    Returns:
        Connectivity status dictionary
    """
    connectivity = {}
    
    # Check HuggingFace official
    try:
        urllib.request.urlopen('https://huggingface.co', timeout=timeout)
        connectivity['huggingface_official'] = True
    except:
        connectivity['huggingface_official'] = False
    
    # Check HuggingFace mirror
    try:
        urllib.request.urlopen('https://hf-mirror.com', timeout=timeout)
        connectivity['huggingface_mirror'] = True
    except:
        connectivity['huggingface_mirror'] = False
    
    # Check ModelScope
    try:
        urllib.request.urlopen('https://modelscope.cn', timeout=timeout)
        connectivity['modelscope'] = True
    except:
        connectivity['modelscope'] = False
    
    return connectivity

def get_directory_size(path: Path) -> int:
    """Get directory size in bytes"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
    except Exception as e:
        print(f"Failed to calculate directory size: {e}")
    return total_size

def list_cached_models(cache_dir: Optional[str] = None) -> List[Dict[str, Union[str, int]]]:
    """
    List locally cached models
    
    Args:
        cache_dir: Custom cache directory, uses default if None
        
    Returns:
        List of cached model information
    """
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    
    cached_models = []
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"Cache directory does not exist: {cache_path}")
        return cached_models
    
    for model_dir in cache_path.iterdir():
        if model_dir.is_dir() and model_dir.name.startswith('models--'):
            # Parse model name
            model_name = model_dir.name.replace('models--', '').replace('--', '/')
            
            # Get model size
            model_size = get_directory_size(model_dir)
            
            # Check model integrity
            is_complete = verify_cache_integrity(model_name, cache_dir)
            
            cached_models.append({
                'model_id': model_name,
                'cache_path': str(model_dir),
                'size_mb': round(model_size / (1024 * 1024), 2),
                'is_complete': is_complete
            })
    
    return cached_models

def verify_cache_integrity(model_id: str, cache_dir: Optional[str] = None) -> bool:
    """
    Verify model cache integrity
    
    Args:
        model_id: Model ID
        cache_dir: Custom cache directory, uses default if None
        
    Returns:
        Whether cache is complete
    """
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    
    try:
        cache_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_files_only=True,
            allow_patterns=["model_index.json"]
        )
        
        # Check required files
        required_files = [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "tokenizer/tokenizer_config.json",
            "unet/config.json",
            "vae/config.json"
        ]
        
        for file_path in required_files:
            full_path = Path(cache_path) / file_path
            if not full_path.exists():
                print(f"Missing file: {full_path}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Failed to verify cache integrity: {e}")
        return False

def cleanup_incomplete_cache(model_id: str, cache_dir: Optional[str] = None) -> bool:
    """
    Clean up incomplete cache
    
    Args:
        model_id: Model ID
        cache_dir: Custom cache directory, uses default if None
        
    Returns:
        Whether cleanup was successful
    """
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    
    try:
        # Convert model_id to cache directory name
        cache_dir_name = f"models--{model_id.replace('/', '--')}"
        cache_path = Path(cache_dir) / cache_dir_name
        
        if cache_path.exists():
            print(f"Cleaning cache directory: {cache_path}")
            shutil.rmtree(cache_path)
            return True
        else:
            print(f"Cache directory does not exist: {cache_path}")
            return False
            
    except Exception as e:
        print(f"Failed to cleanup cache: {e}")
        return False

def cleanup_all_cache(cache_dir: Optional[str] = None) -> bool:
    """
    Clean up all cache
    
    Args:
        cache_dir: Custom cache directory, uses default if None
        
    Returns:
        Whether cleanup was successful
    """
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    
    try:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            print(f"Cleaning entire cache directory: {cache_path}")
            shutil.rmtree(cache_path)
            cache_path.mkdir(parents=True, exist_ok=True)
            return True
        return True
    except Exception as e:
        print(f"Failed to cleanup all cache: {e}")
        return False

def download_model(model_id: str, cache_dir: Optional[str] = None, force_download: bool = False) -> Optional[str]:
    """
    Download model to local cache
    
    Args:
        model_id: Model ID
        cache_dir: Custom cache directory, uses default if None
        force_download: Whether to force re-download
        
    Returns:
        Cache path, None if failed
    """
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    
    try:
        print(f"Downloading model: {model_id}")
        
        # Check network connectivity
        connectivity = check_network_connectivity()
        if not any(connectivity.values()):
            print("Network unreachable, cannot download model")
            return None
        
        # Set mirror if official site is unreachable
        if not connectivity['huggingface_official'] and connectivity['huggingface_mirror']:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        cache_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=True
        )
        
        print(f"Model download completed: {cache_path}")
        return cache_path
        
    except Exception as e:
        print(f"Failed to download model: {e}")
        return None

def load_model_offline(model_id: str, cache_dir: Optional[str] = None, **kwargs) -> Optional[DiffusionPipeline]:
    """
    Load model offline
    
    Args:
        model_id: Model ID
        cache_dir: Custom cache directory, uses default if None
        **kwargs: Additional arguments for DiffusionPipeline.from_pretrained
        
    Returns:
        DiffusionPipeline instance, None if failed
    """
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    
    try:
        print(f"Loading model offline: {model_id}")
        
        if not verify_cache_integrity(model_id, cache_dir):
            print(f"Model cache incomplete: {model_id}")
            return None
        
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=True,
            **kwargs
        )
        
        print(f"Model loaded successfully: {model_id}")
        return pipeline
        
    except Exception as e:
        print(f"Failed to load model offline: {e}")
        return None

def get_model_info(model_id: str) -> Optional[Dict]:
    """
    Get model information
    
    Args:
        model_id: Model ID
        
    Returns:
        Model information dictionary
    """
    try:
        api = HfApi()
        model_info = api.model_info(model_id)
        return {
            'model_id': model_id,
            'downloads': getattr(model_info, 'downloads', 0),
            'likes': getattr(model_info, 'likes', 0),
            'tags': getattr(model_info, 'tags', []),
            'pipeline_tag': getattr(model_info, 'pipeline_tag', None),
            'library_name': getattr(model_info, 'library_name', None),
            'created_at': str(getattr(model_info, 'created_at', '')),
            'last_modified': str(getattr(model_info, 'last_modified', ''))
        }
    except Exception as e:
        print(f"Failed to get model info: {e}")
        return None

def get_cache_stats(cache_dir: Optional[str] = None) -> Dict[str, Union[int, float]]:
    """
    Get cache statistics
    
    Args:
        cache_dir: Custom cache directory, uses default if None
        
    Returns:
        Cache statistics dictionary
    """
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return {'total_models': 0, 'total_size_mb': 0.0}
    
    total_size = get_directory_size(cache_path)
    model_count = len([d for d in cache_path.iterdir() if d.is_dir() and d.name.startswith('models--')])
    
    return {
        'total_models': model_count,
        'total_size_mb': round(total_size / (1024 * 1024), 2),
        'cache_path': str(cache_path)
    }

# Convenience functions
def quick_model_check(model_id: str, cache_dir: Optional[str] = None) -> Dict[str, bool]:
    """
    Quick model status check
    
    Args:
        model_id: Model ID
        cache_dir: Cache directory
        
    Returns:
        Status dictionary
    """
    return {
        'cached': verify_cache_integrity(model_id, cache_dir),
        'network_available': any(check_network_connectivity().values())
    }

def setup_china_mirror():
    """Setup China mirror source"""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print("HuggingFace China mirror source configured")

def setup_offline_mode():
    """Setup offline mode"""
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    print("Offline mode configured")