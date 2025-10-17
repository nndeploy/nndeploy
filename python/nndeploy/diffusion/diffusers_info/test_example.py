
import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from diffusers.utils import logging

logging.set_verbosity_info()

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
)
pipe.enable_sequential_cpu_offload()
# pipe.to("cuda")
prompt = "A cat holding a sign that says hello world"
image = pipe(prompt).images[0]
image.save("sd3.png")

# import torch
# from diffusers import CogView4Pipeline, AutoPipelineForText2Image
# print("CogView4Pipeline")
# pipe = AutoPipelineForText2Image.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16)
# pipe.enable_sequential_cpu_offload()
# print("CogView4Pipeline loaded")
# # pipe.to("cuda")
# prompt = "A photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]
# image.save("output.png")