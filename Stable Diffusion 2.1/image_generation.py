# Stable Diffusion 2.1 interactive prompt script for PyCharm
from diffusers import StableDiffusionPipeline
import torch
import pathlib
import re
from PIL import Image as PILImage

# Define your desired output folder:
# Adjust this to your project path
output_dir = pathlib.Path(r"C:\Users\sanji\PycharmProjects\text-to-3d\Stable Diffusion 2.1\SD Output")
output_dir.mkdir(parents=True, exist_ok=True)

# Load Stable Diffusion 2.1 model
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

pipe.enable_attention_slicing()

print("Stable Diffusion 2.1 pipeline loaded.")
print(f"Using device: {device}")
print(f"Images will be saved to: {output_dir}")

# Interactive loop
while True:
    # Get user prompt
    prompt = input("Enter your image prompt (or type 'exit' to stop): ")
    if prompt.lower() == "exit":
        print("Exiting...")
        break

    # Sanitize filename
    safe_filename = re.sub(r'[^a-zA-Z0-9_\-]', '_', prompt)[:50]

    # Generate image
    print(f"Generating image for: '{prompt}'...")
    image = pipe(prompt, height=512, width=512, num_inference_steps=30).images[0]

    # Save image
    filename = output_dir / f"{safe_filename}.png"
    image.save(filename)

    print(f"Image saved to: {filename}")

    image.show(title=f"{prompt}")
