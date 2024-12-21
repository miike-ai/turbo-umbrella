import os
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from pruna import SmashConfig, smash

import torch

torch._dynamo.config.cache_size_limit = 64  # Increase the cache size to reduce recompilation
torch._dynamo.config.suppress_errors = True  # Suppress errors from Dynamo
torch.set_float32_matmul_precision("high")  # Ensure high precision for float32 matrix multiplications
torch._dynamo.config.dynamic_shapes = False


# Environment variables
os.environ["PYTORCH_ENABLE_WARNINGS"] = "0"  # Suppress warnings
os.environ["TORCH_DYNAMO_CACHE_SIZE"] = "64"  # Increase cache size

# Load the Flux Model
print("Loading the Flux model...")
pipe = FluxPipeline.from_pretrained("miike-ai/skittles-v2", torch_dtype=torch.bfloat16)
pipe.to('cuda')

# Compile the model with TorchDynamo and Inductor
# pipe = torch.compile(pipe, backend="inductor")

# Initialize Smash Config
print("Initializing Smash Config...")
smash_config = SmashConfig()
smash_config['compilers'] = ['flux_caching']
smash_config['comp_flux_caching_cache_interval'] = 2
smash_config['comp_flux_caching_start_step'] = 2
smash_config['comp_flux_caching_compile'] = True
smash_config['comp_flux_caching_save_model'] = False

# Smash the Model
print("Smashing the model...")
pruna_token = os.getenv("PRUNA_TOKEN")
if not pruna_token:
    raise ValueError("PRUNA_TOKEN environment variable is not set. Please set it and try again.")

pipe = smash(
    model=pipe,
    token=pruna_token,
    smash_config=smash_config,
)

# Run inference

pipe.scheduler.set_timesteps(50)  # Ensure consistent timesteps
pipe.to(torch.float16)  # Use FP16 for faster inference

print("Running inference...")
prompt = "A red apple"
num_inference_steps = 50
width = 512
height = 512
image = pipe(prompt, num_inference_steps=num_inference_steps, width=width, height=height).images[0]
output_filename = "output_image.png"
image.save(output_filename)

print(f"Image generated and saved as {output_filename}.")
