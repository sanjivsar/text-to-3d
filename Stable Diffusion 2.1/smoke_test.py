from diffusers import StableDiffusionPipeline
import torch, pathlib

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    torch_dtype=torch.float16
).to("cuda")

pipe.enable_attention_slicing()

prompt = "a futuristic neon tiger, concept art, 4k"
image = pipe(prompt, height=512, width=512).images[0]

pathlib.Path("outputs").mkdir(exist_ok=True)
image.save("outputs/tiger.png")


