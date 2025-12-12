from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained(
    "./sdxl-turbo",
    torch_dtype=torch.float32,
).to("cpu")

image_prompt = "a car in beach sunset"

image = pipe(image_prompt).images[0]
image.save("generated.png")