
from .pipelines_dict import DIFFUSERS_PIPELINES_DICT

def get_text2image_pipelines_pretrained_model_paths(supports_auto_pipeline: bool = True):
    """
    获取所有支持“文本生成图像（text2image）”功能的pipeline的预训练模型路径。

    主要用途：
    - 用于根据文本描述生成对应的图像，适用于AI绘画、文生图等场景。
    - 可筛选是否支持AutoPipeline机制。

    参数:
        supports_auto_pipeline (bool): 是否只返回支持AutoPipeline的pipeline。

    返回:
        List[str]: 所有pipeline的预训练模型路径的单列表。
    """
    # 先用set去重，并移除以"example/"开头的路径，最后转为list返回
    return list({
        path
        for pipeline in DIFFUSERS_PIPELINES_DICT.values()
        if pipeline["function"] == "text2image" and pipeline["supports_auto_pipeline"] == supports_auto_pipeline
        for path in pipeline["pretrained_model_paths"]
        if not path.startswith("example/")
    })

def get_image2image_pipelines_pretrained_model_paths(supports_auto_pipeline: bool = True):
    """
    获取所有支持“图像到图像（image2image）”功能的pipeline的预训练模型路径。

    主要用途：
    - 用于对输入图像进行风格迁移、重绘、修复等处理，输入输出均为图像。
    - 可筛选是否支持AutoPipeline机制。

    参数:
        supports_auto_pipeline (bool): 是否只返回支持AutoPipeline的pipeline。

    返回:
        List[str]: 所有pipeline的预训练模型路径的单列表。
    """
    # 用set去重，并移除以"example/"开头的路径，最后转为list返回
    return list({
        path
        for pipeline in DIFFUSERS_PIPELINES_DICT.values()
        if pipeline["function"] == "image2image" and pipeline["supports_auto_pipeline"] == supports_auto_pipeline
        for path in pipeline["pretrained_model_paths"]
        if not path.startswith("example/")
    })

def get_inpainting_pipelines_pretrained_model_paths(supports_auto_pipeline: bool = True):
    """
    获取所有支持“图像修复（inpainting）”功能的pipeline的预训练模型路径。

    主要用途：
    - 用于对图像中指定区域进行智能修复、补全、去除等操作。
    - 适合图片内容填充、去水印等场景。
    - 可筛选是否支持AutoPipeline机制。

    参数:
        supports_auto_pipeline (bool): 是否只返回支持AutoPipeline的pipeline。

    返回:
        List[str]: 所有pipeline的预训练模型路径的单列表。
    """
    return list({
        path
        for pipeline in DIFFUSERS_PIPELINES_DICT.values()
        if pipeline["function"] == "inpainting" and pipeline["supports_auto_pipeline"] == supports_auto_pipeline
        for path in pipeline["pretrained_model_paths"]
        if not path.startswith("example/")
    })

def get_multi_modal_pipelines_pretrained_model_paths(supports_auto_pipeline: bool = True):
    """
    获取所有支持“多模态（multi_modal）”功能的pipeline的预训练模型路径。

    主要用途：
    - 用于处理多种输入模态（如文本、图像、音频等）的生成或理解任务。
    - 适合跨模态检索、生成等复杂场景。
    - 可筛选是否支持AutoPipeline机制。

    参数:
        supports_auto_pipeline (bool): 是否只返回支持AutoPipeline的pipeline。

    返回:
        List[str]: 所有pipeline的预训练模型路径的单列表。
    """
    return list({
        path
        for pipeline in DIFFUSERS_PIPELINES_DICT.values()
        if pipeline["function"] == "multi_modal" and pipeline["supports_auto_pipeline"] == supports_auto_pipeline
        for path in pipeline["pretrained_model_paths"]
        if not path.startswith("example/")
    })

def get_text2video_pipelines_pretrained_model_paths(supports_auto_pipeline: bool = True):
    """
    获取所有支持“文本生成视频（text2video）”功能的pipeline的预训练模型路径。

    主要用途：
    - 用于根据文本描述生成对应的视频内容，适合AI视频生成、文生视频等场景。
    - 可筛选是否支持AutoPipeline机制。

    参数:
        supports_auto_pipeline (bool): 是否只返回支持AutoPipeline的pipeline。

    返回:
        List[str]: 所有pipeline的预训练模型路径的单列表。
    """
    return list({
        path
        for pipeline in DIFFUSERS_PIPELINES_DICT.values()
        if pipeline["function"] == "text2video" and pipeline["supports_auto_pipeline"] == supports_auto_pipeline
        for path in pipeline["pretrained_model_paths"]
        if not path.startswith("example/")
    })

def get_video2video_pipelines_pretrained_model_paths(supports_auto_pipeline: bool = True):
    """
    获取所有支持“视频到视频（video2video）”功能的pipeline的预训练模型路径。

    主要用途：
    - 用于对输入视频进行风格迁移、内容增强、视频修复等处理，输入输出均为视频。
    - 可筛选是否支持AutoPipeline机制。

    参数:
        supports_auto_pipeline (bool): 是否只返回支持AutoPipeline的pipeline。

    返回:
        List[str]: 所有pipeline的预训练模型路径的单列表。
    """
    return list({
        path
        for pipeline in DIFFUSERS_PIPELINES_DICT.values()
        if pipeline["function"] == "video2video" and pipeline["supports_auto_pipeline"] == supports_auto_pipeline
        for path in pipeline["pretrained_model_paths"]
        if not path.startswith("example/")
    })

def get_image2video_pipelines_pretrained_model_paths(supports_auto_pipeline: bool = True):
    """
    获取所有支持“图像生成视频（image2video）”功能的pipeline的预训练模型路径。

    主要用途：
    - 用于将静态图像扩展为动态视频，适合动画生成、图片动化等场景。
    - 可筛选是否支持AutoPipeline机制。

    参数:
        supports_auto_pipeline (bool): 是否只返回支持AutoPipeline的pipeline。

    返回:
        List[str]: 所有pipeline的预训练模型路径的单列表。
    """
    return list({
        path
        for pipeline in DIFFUSERS_PIPELINES_DICT.values()
        if pipeline["function"] == "image2video" and pipeline["supports_auto_pipeline"] == supports_auto_pipeline
        for path in pipeline["pretrained_model_paths"]
        if not path.startswith("example/")
    })

def get_unconditional_image_pipelines_pretrained_model_paths(supports_auto_pipeline: bool = True):
    """
    获取所有支持“无条件图像生成（unconditional_image）”功能的pipeline的预训练模型路径。

    主要用途：
    - 用于不依赖任何输入条件（如文本、图像等）直接生成图像，适合纯噪声生成、艺术创作等场景。
    - 可筛选是否支持AutoPipeline机制。

    参数:
        supports_auto_pipeline (bool): 是否只返回支持AutoPipeline的pipeline。

    返回:
        List[str]: 所有pipeline的预训练模型路径的单列表。
    """
    return list({
        path
        for pipeline in DIFFUSERS_PIPELINES_DICT.values()
        if pipeline["function"] == "unconditional_image" and pipeline["supports_auto_pipeline"] == supports_auto_pipeline
        for path in pipeline["pretrained_model_paths"]
        if not path.startswith("example/")
    })

def get_unconditional_text_pipelines_pretrained_model_paths(supports_auto_pipeline: bool = True):
    """
    获取所有支持“无条件文本生成（unconditional_text）”功能的pipeline的预训练模型路径。

    主要用途：
    - 用于不依赖任何输入条件直接生成文本内容，适合文本生成、语言建模等场景。
    - 可筛选是否支持AutoPipeline机制。

    参数:
        supports_auto_pipeline (bool): 是否只返回支持AutoPipeline的pipeline。

    返回:
        List[str]: 所有pipeline的预训练模型路径的单列表。
    """
    return list({
        path
        for pipeline in DIFFUSERS_PIPELINES_DICT.values()
        if pipeline["function"] == "unconditional_text" and pipeline["supports_auto_pipeline"] == supports_auto_pipeline
        for path in pipeline["pretrained_model_paths"]
        if not path.startswith("example/")
    })

def get_unconditional_audio_pipelines_pretrained_model_paths(supports_auto_pipeline: bool = True):
    """
    获取所有支持“无条件音频生成（unconditional_audio）”功能的pipeline的预训练模型路径。

    主要用途：
    - 用于不依赖任何输入条件直接生成音频内容，适合音乐生成、音效合成等场景。
    - 可筛选是否支持AutoPipeline机制。

    参数:
        supports_auto_pipeline (bool): 是否只返回支持AutoPipeline的pipeline。

    返回:
        List[str]: 所有pipeline的预训练模型路径的单列表。
    """
    return list({
        path
        for pipeline in DIFFUSERS_PIPELINES_DICT.values()
        if pipeline["function"] == "unconditional_audio" and pipeline["supports_auto_pipeline"] == supports_auto_pipeline
        for path in pipeline["pretrained_model_paths"]
        if not path.startswith("example/")
    })