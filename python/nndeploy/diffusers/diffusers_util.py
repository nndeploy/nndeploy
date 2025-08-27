
import os
import shutil
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional, Union

from diffusers import DiffusionPipeline
from huggingface_hub import list_models, snapshot_download, HfApi
from huggingface_hub.utils import HfHubHTTPError

class DiffusersUtil:
    """Diffusers utility class providing model management and cache operations"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize utility class
        
        Args:
            cache_dir: Custom cache directory, uses default if None
        """
        self.cache_dir = cache_dir or self._get_default_cache_dir()
        self.api = HfApi()
    
    @staticmethod
    def _get_default_cache_dir() -> str:
        """Get default cache directory"""
        hf_home = os.environ.get('HF_HOME')
        if hf_home:
            return os.path.join(hf_home, 'hub')
        
        hf_cache = os.environ.get('HUGGINGFACE_HUB_CACHE')
        if hf_cache:
            return hf_cache
        
        return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    
    def get_supported_models(self, library: str = "diffusers", limit: int = 100) -> List[str]:
        """
        Get list of supported diffusers models
        
        Args:
            library: Library name, defaults to "diffusers"
            limit: Limit of returned model count
            
        Returns:
            List of model IDs
        """
        # try:
        #     print(f"Getting {library} supported model list...")
        #     models = list_models(library=library, limit=limit)
        #     model_ids = [model.modelId for model in models]
        #     print(f"Found {len(model_ids)} models")
        #     return model_ids
        # except Exception as e:
        #     print(f"Failed to get model list: {e}")
        #     # Return some common stable diffusion models as fallback
        #     return [
        #         "runwayml/stable-diffusion-v1-5",
        #         "stable-diffusion-v1-5/stable-diffusion-v1-5",
        #         "stabilityai/stable-diffusion-xl-base-1.0",
        #         "stabilityai/stable-diffusion-2-1",
        #         "CompVis/stable-diffusion-v1-4"
        #     ]
        # 基于 Hugging Face Diffusers 官方文档支持的主流模型列表
        # 参考: https://github.com/huggingface/diffusers
        return [
            # Stable Diffusion v1.x 系列
            "runwayml/stable-diffusion-v1-5",
            "stable-diffusion-v1-5/stable-diffusion-v1-5", 
            "CompVis/stable-diffusion-v1-4",
            
            # Stable Diffusion v2.x 系列
            "stabilityai/stable-diffusion-2-1",
            "stabilityai/stable-diffusion-2-base",
            
            # Stable Diffusion XL 系列
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            
            # 图像修复 (Inpainting) 模型
            "runwayml/stable-diffusion-inpainting",
            "stabilityai/stable-diffusion-2-inpainting",
            
            # 图像超分辨率模型
            "stabilityai/stable-diffusion-x4-upscaler",
            "stabilityai/sd-x2-latent-upscaler",
            
            # ControlNet 控制模型
            "lllyasviel/sd-controlnet-canny",
            "lllyasviel/sd-controlnet-depth",
            "lllyasviel/sd-controlnet-openpose",
            
            # 图像变换模型
            "lambdalabs/sd-image-variations-diffusers",
            "timbrooks/instruct-pix2pix",
            
            # DeepFloyd IF 系列
            "DeepFloyd/IF-I-XL-v1.0",
            
            # Kandinsky 系列
            "kandinsky-community/kandinsky-2-2-decoder",
            
            # unCLIP 系列
            "kakaobrain/karlo-v1-alpha",
            
            # DDPM 经典模型
            "google/ddpm-ema-church-256",
            "google/ddpm-cat-256"
        ]
    
    def get_popular_models(self) -> Dict[str, List[str]]:
        """
        Get categorized popular models list
        
        Returns:
            Dictionary of models grouped by category
        """
        return {
            "stable_diffusion_v1": [
                "runwayml/stable-diffusion-v1-5",
                "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "CompVis/stable-diffusion-v1-4"
            ],
            "stable_diffusion_v2": [
                "stabilityai/stable-diffusion-2-1",
                "stabilityai/stable-diffusion-2-base"
            ],
            "stable_diffusion_xl": [
                "stabilityai/stable-diffusion-xl-base-1.0",
                "stabilityai/stable-diffusion-xl-refiner-1.0"
            ],
            "controlnet": [
                "lllyasviel/sd-controlnet-canny",
                "lllyasviel/sd-controlnet-depth",
                "lllyasviel/sd-controlnet-openpose"
            ],
            "inpainting": [
                "runwayml/stable-diffusion-inpainting",
                "stabilityai/stable-diffusion-2-inpainting"
            ]
        }
    
    def check_network_connectivity(self, timeout: int = 10) -> Dict[str, bool]:
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
    
    def list_cached_models(self) -> List[Dict[str, Union[str, int]]]:
        """
        List locally cached models
        
        Returns:
            List of cached model information
        """
        cached_models = []
        cache_path = Path(self.cache_dir)
        
        if not cache_path.exists():
            print(f"Cache directory does not exist: {cache_path}")
            return cached_models
        
        for model_dir in cache_path.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith('models--'):
                # Parse model name
                model_name = model_dir.name.replace('models--', '').replace('--', '/')
                
                # Get model size
                model_size = self._get_directory_size(model_dir)
                
                # Check model integrity
                is_complete = self.verify_cache_integrity(model_name)
                
                cached_models.append({
                    'model_id': model_name,
                    'cache_path': str(model_dir),
                    'size_mb': round(model_size / (1024 * 1024), 2),
                    'is_complete': is_complete
                })
        
        return cached_models
    
    def verify_cache_integrity(self, model_id: str) -> bool:
        """
        Verify model cache integrity
        
        Args:
            model_id: Model ID
            
        Returns:
            Whether cache is complete
        """
        try:
            cache_path = snapshot_download(
                repo_id=model_id,
                cache_dir=self.cache_dir,
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
    
    def cleanup_incomplete_cache(self, model_id: str) -> bool:
        """
        Clean up incomplete cache
        
        Args:
            model_id: Model ID
            
        Returns:
            Whether cleanup was successful
        """
        try:
            # Convert model_id to cache directory name
            cache_dir_name = f"models--{model_id.replace('/', '--')}"
            cache_path = Path(self.cache_dir) / cache_dir_name
            
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
    
    def cleanup_all_cache(self) -> bool:
        """
        Clean up all cache
        
        Returns:
            Whether cleanup was successful
        """
        try:
            cache_path = Path(self.cache_dir)
            if cache_path.exists():
                print(f"Cleaning entire cache directory: {cache_path}")
                shutil.rmtree(cache_path)
                cache_path.mkdir(parents=True, exist_ok=True)
                return True
            return True
        except Exception as e:
            print(f"Failed to cleanup all cache: {e}")
            return False
    
    def download_model(self, model_id: str, force_download: bool = False) -> Optional[str]:
        """
        Download model to local cache
        
        Args:
            model_id: Model ID
            force_download: Whether to force re-download
            
        Returns:
            Cache path, None if failed
        """
        try:
            print(f"Downloading model: {model_id}")
            
            # Check network connectivity
            connectivity = self.check_network_connectivity()
            if not any(connectivity.values()):
                print("Network unreachable, cannot download model")
                return None
            
            # Set mirror if official site is unreachable
            if not connectivity['huggingface_official'] and connectivity['huggingface_mirror']:
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            
            cache_path = snapshot_download(
                repo_id=model_id,
                cache_dir=self.cache_dir,
                force_download=force_download,
                resume_download=True
            )
            
            print(f"Model download completed: {cache_path}")
            return cache_path
            
        except Exception as e:
            print(f"Failed to download model: {e}")
            return None
    
    def load_model_offline(self, model_id: str, **kwargs) -> Optional[DiffusionPipeline]:
        """
        Load model offline
        
        Args:
            model_id: Model ID
            **kwargs: Additional arguments for DiffusionPipeline.from_pretrained
            
        Returns:
            DiffusionPipeline instance, None if failed
        """
        try:
            print(f"Loading model offline: {model_id}")
            
            if not self.verify_cache_integrity(model_id):
                print(f"Model cache incomplete: {model_id}")
                return None
            
            pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                cache_dir=self.cache_dir,
                local_files_only=True,
                **kwargs
            )
            
            print(f"Model loaded successfully: {model_id}")
            return pipeline
            
        except Exception as e:
            print(f"Failed to load model offline: {e}")
            return None
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """
        Get model information
        
        Args:
            model_id: Model ID
            
        Returns:
            Model information dictionary
        """
        try:
            model_info = self.api.model_info(model_id)
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
    
    def set_cache_dir(self, cache_dir: str):
        """
        Set cache directory
        
        Args:
            cache_dir: New cache directory path
        """
        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"Cache directory set to: {cache_dir}")
    
    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get cache statistics
        
        Returns:
            Cache statistics dictionary
        """
        cache_path = Path(self.cache_dir)
        if not cache_path.exists():
            return {'total_models': 0, 'total_size_mb': 0.0}
        
        total_size = self._get_directory_size(cache_path)
        model_count = len([d for d in cache_path.iterdir() if d.is_dir() and d.name.startswith('models--')])
        
        return {
            'total_models': model_count,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_path': str(cache_path)
        }
    
    @staticmethod
    def _get_directory_size(path: Path) -> int:
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

# Convenience functions
def get_diffusers_util(cache_dir: Optional[str] = None) -> DiffusersUtil:
    """Get DiffusersUtil instance"""
    return DiffusersUtil(cache_dir)

def quick_model_check(model_id: str, cache_dir: Optional[str] = None) -> Dict[str, bool]:
    """
    Quick model status check
    
    Args:
        model_id: Model ID
        cache_dir: Cache directory
        
    Returns:
        Status dictionary
    """
    util = DiffusersUtil(cache_dir)
    return {
        'cached': util.verify_cache_integrity(model_id),
        'network_available': any(util.check_network_connectivity().values())
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


# # Create utility instance
# util = DiffusersUtil(cache_dir="/custom/cache/path")

# # Check network connectivity
# connectivity = util.check_network_connectivity()
# print(connectivity)

# # Get supported models
# models = util.get_supported_models(limit=20)
# print(models)

# # Check local cache
# cached_models = util.list_cached_models()
# print(cached_models)

# # Download model
# cache_path = util.download_model("runwayml/stable-diffusion-v1-5")

# # Load model offline
# pipeline = util.load_model_offline("runwayml/stable-diffusion-v1-5")