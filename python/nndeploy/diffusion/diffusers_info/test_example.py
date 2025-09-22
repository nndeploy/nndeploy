
import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
)
pipe.to("cuda")
prompt = "A cat holding a sign that says hello world"
image = pipe(prompt).images[0]
image.save("sd3.png")