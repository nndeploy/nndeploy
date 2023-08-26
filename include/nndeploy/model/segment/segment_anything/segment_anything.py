
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import requests
from transformers import SamModel, SamProcessor
from transformers import pipeline


class SegmentAnything:
    def __init__(self) -> None:
        model = SamModel.from_pretrained("facebook/sam-vit-base")
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        input_points = [[[450, 600]]] # 2D localization of a window

    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    def transformers_infer(self):
        inputs = self.processor(self.raw_image, input_points=self.input_points, return_tensors="pt").to("cuda")
        outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        scores = outputs.iou_scores

        generator =  pipeline("mask-generation", device = 0, points_per_batch = 256)
        image_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
        outputs = generator(image_url, points_per_batch = 256)

        plt.imshow(np.array(self.raw_image))
        ax = plt.gca()
        for mask in outputs["masks"]:
            self.show_mask(mask, ax=ax, random_color=True)
        plt.axis("off")
        plt.show()
                

    def onnxruntime_infer(self):
        pass
    
    def export_torchscript(self, path: str) -> None:
        pass
    
    def export_tf(self, path: str) -> None:
        pass    
    
    def export_onnx(self, path: str) -> None:
        pass
        

def main():
    pass


if __name__ == "__main__":
    main()