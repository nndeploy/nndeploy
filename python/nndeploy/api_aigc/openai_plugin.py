import json
import logging
import base64
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import os

import nndeploy.dag
import nndeploy.base


class OpenAIImageNode(nndeploy.dag.Node):
    """OpenAI图像生成节点（仅保留此节点）"""

    def __init__(self, name, inputs: list[nndeploy.dag.Edge] = None, outputs: list[nndeploy.dag.Edge] = None):
        super().__init__(name, inputs, outputs)
        super().set_key("nndeploy.openai.OpenAIImageNode")
        super().set_desc("OpenAI图像生成节点 - 支持DALL-E图像生成")
        self._logger = logging.getLogger(__name__)

        # 设置输入输出类型
        self.set_input_type(str)  # 输入图像描述
        self.set_output_type(np.ndarray)  # 输出生成的图像

        # 前端可配置的参数
        self.api_key = ""  # API密钥
        self.base_url = "https://api.openai.com/v1"  # 基础URL（可改为其他兼容服务）
        self.model = "dall-e-3"  # 模型名称
        self.size = "1024x1024"  # 图像尺寸
        self.quality = "standard"  # 图像质量
        self.style = "vivid"  # 图像风格

    def run(self):
        try:
            # 获取输入
            input_edge = self.get_input(0)
            prompt = input_edge.get(self)
            self._logger.info("[OpenAIImageNode] Received prompt: %s",
                              (prompt[:80] + '...') if isinstance(prompt, str) and len(prompt) > 80 else prompt)

            if not self.api_key:
                self._logger.error("[OpenAIImageNode] API key not set")
                return nndeploy.base.Status.error("OpenAI API密钥未设置")

            # 构建请求
            self._logger.debug("[OpenAIImageNode] Building request | base_url=%s model=%s size=%s quality=%s style=%s",
                               self.base_url, self.model, self.size, self.quality, self.style)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model,
                "prompt": prompt,
                "size": self.size,
                "quality": self.quality,
                "style": self.style,
                "n": 1
            }

            # 发送请求（使用可配置base_url）
            url = f"{self.base_url.rstrip('/')}/images/generations"
            self._logger.info("[OpenAIImageNode] POST %s", url)
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=60
            )
            self._logger.info("[OpenAIImageNode] Response status: %s", response.status_code)

            if response.status_code == 200:
                result = response.json()
                self._logger.debug("[OpenAIImageNode] Response JSON keys: %s", list(result.keys()))
                # 优先读取b64_json，其次读取url（兼容不同服务实现）
                image_b64 = None
                image_url = None
                if isinstance(result.get("data"), list) and result["data"]:
                    item = result["data"][0]
                    image_b64 = item.get("b64_json")
                    image_url = item.get("url")
                self._logger.debug("[OpenAIImageNode] Have b64:%s url:%s", bool(image_b64), bool(image_url))

                if image_b64:
                    img = Image.open(BytesIO(base64.b64decode(image_b64)))
                    img_array = np.array(img)
                elif image_url:
                    self._logger.info("[OpenAIImageNode] Downloading image: %s", image_url)
                    img_response = requests.get(image_url, timeout=30)
                    if img_response.status_code != 200:
                        self._logger.error("[OpenAIImageNode] Image download failed: %s", img_response.status_code)
                        return nndeploy.base.Status.error("图像下载失败")
                    img = Image.open(BytesIO(img_response.content))
                    img_array = np.array(img)
                else:
                    self._logger.error("[OpenAIImageNode] No image data in response")
                    return nndeploy.base.Status.error("返回结果中未包含图像数据")

                # 保存到文件
                try:
                    save_dir = os.path.join("resources", "images")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "result.openai.jpg")
                    pil_to_save = Image.fromarray(img_array)
                    if pil_to_save.mode in ("RGBA", "P"):
                        pil_to_save = pil_to_save.convert("RGB")
                    pil_to_save.save(save_path, format="JPEG")
                    self._logger.info("[OpenAIImageNode] Image saved to %s", save_path)
                except Exception:
                    self._logger.exception("[OpenAIImageNode] Failed to save image to file")

                # 设置输出
                output_edge = self.get_output(0)
                output_edge.set(img_array)
                self._logger.info("[OpenAIImageNode] Output image set: shape=%s dtype=%s",
                                  getattr(img_array, 'shape', None), getattr(img_array, 'dtype', None))
                return nndeploy.base.Status.ok()
            else:
                error_msg = f"图像生成API请求失败: {response.status_code} - {response.text}"
                self._logger.error("[OpenAIImageNode] %s", error_msg)
                return nndeploy.base.Status.error(error_msg)

        except Exception as e:
            self._logger.exception("[OpenAIImageNode] Exception during run")
            return nndeploy.base.Status.error(f"图像生成错误: {str(e)}")

    def serialize(self):
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj.update({
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "size": self.size,
            "quality": self.quality,
            "style": self.style
        })
        return json.dumps(json_obj)

    def deserialize(self, target: str):
        json_obj = json.loads(target)
        self.api_key = json_obj.get("api_key", "")
        self.base_url = json_obj.get("base_url", "https://api.openai.com/v1")
        self.model = json_obj.get("model", "dall-e-3")
        self.size = json_obj.get("size", "1024x1024")
        self.quality = json_obj.get("quality", "standard")
        self.style = json_obj.get("style", "vivid")
        return super().deserialize(target)


class OpenAIImageNodeCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()

    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = OpenAIImageNode(name, inputs, outputs)
        return self.node


# 注册节点
openai_image_node_creator = OpenAIImageNodeCreator()
nndeploy.dag.register_node("nndeploy.api_aigc.OpenAIImageNode", openai_image_node_creator)


