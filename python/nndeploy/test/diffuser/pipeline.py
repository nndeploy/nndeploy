from diffusers import DiffusionPipeline
import torch
import time

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)

start_time = time.time()
pipeline.to("cuda")
end_time = time.time()
print(f"Time taken to load pipeline: {end_time - start_time} seconds")

start_time = time.time()
print(pipeline("An image of a squirrel in Picasso style").images[0])
end_time = time.time()
print(f"Time taken to generate image: {end_time - start_time} seconds")