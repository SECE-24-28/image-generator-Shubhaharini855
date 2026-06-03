from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

# Load model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# User input
prompt = input("Enter image prompt: ")
filename = input("Enter output filename (without extension): ")

print("Generating image... Please wait.")

# Generate image
if device == "cuda":
    with torch.autocast("cuda"):
        image = pipe(prompt).images[0]
else:
    image = pipe(prompt).images[0]

# Display
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis("off")
plt.show()

# Save
image.save(f"{filename}.png")

print(f"Image saved successfully as {filename}.png")