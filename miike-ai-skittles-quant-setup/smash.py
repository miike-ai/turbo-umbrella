import os
import torch
from diffusers import FluxPipeline
from pruna import SmashConfig, smash

# Step 1: Load the Flux Model
print("Loading the Flux model...")
pipe = FluxPipeline.from_pretrained("miike-ai/skittles-v2", torch_dtype=torch.bfloat16)
pipe.to('cuda')  # Move the model to GPU
# Uncomment the following line to offload some parts of the model to CPU if needed
# pipe.enable_model_cpu_offload()

print("Model loaded and moved to GPU.")

# Step 2: Initialize the Smash Config
print("Initializing Smash Config...")
smash_config = SmashConfig()
smash_config['compilers'] = ['flux_caching']
smash_config['comp_flux_caching_cache_interval'] = 2  # Higher is faster but reduces quality
smash_config['comp_flux_caching_start_step'] = 2  # Should match cache_interval
smash_config['comp_flux_caching_compile'] = True  # Compile for extra speed-up
smash_config['comp_flux_caching_save_model'] = False  # Only use for inference, don't save

print("Smash Config initialized.")

# Step 3: Smash the Model
print("Smashing the model...")
pruna_token = os.getenv("PRUNA_TOKEN")  # Get the Pruna token from the environment variable
if not pruna_token:
    raise ValueError("PRUNA_TOKEN environment variable is not set. Please set it and try again.")

pipe = smash(
    model=pipe,
    token=pruna_token,
    smash_config=smash_config,
)

print("Model smashed successfully.")

# Step 4: Run the Model for Inference
print("Running inference...")
prompt = "A red apple"
num_inference_steps = 4

# Generate the image
image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
output_filename = "output_image.png"
image.save(output_filename)

print(f"Image generated and saved as {output_filename}.")
