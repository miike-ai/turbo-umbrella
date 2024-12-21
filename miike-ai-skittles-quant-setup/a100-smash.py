import os
import torch
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from pruna import SmashConfig, smash

# Environment variable for Pruna token
PRUNA_TOKEN = os.getenv("PRUNA_TOKEN")
if not PRUNA_TOKEN:
    raise ValueError("PRUNA_TOKEN environment variable is not set. Please set it and try again.")

# Model configuration
model_id = "miike-ai/skittles-v2"
model_revision = "main"
text_model_id = "openai/clip-vit-large-patch14"
model_data_type = torch.bfloat16

# Step 1: Load the Flux Model Components
print("Loading model components...")

# Load CLIP tokenizer and text encoder
tokenizer = CLIPTokenizer.from_pretrained(text_model_id, torch_dtype=model_data_type)
text_encoder = CLIPTextModel.from_pretrained(text_model_id, torch_dtype=model_data_type)

# Load T5 tokenizer and text encoder
tokenizer_2 = T5TokenizerFast.from_pretrained(
    model_id, subfolder="tokenizer_2", torch_dtype=model_data_type, revision=model_revision
)
text_encoder_2 = T5EncoderModel.from_pretrained(
    model_id, subfolder="text_encoder_2", torch_dtype=model_data_type, revision=model_revision
)

# Load scheduler
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    model_id, subfolder="scheduler", revision=model_revision
)

# Load transformer
transformer = FluxTransformer2DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=model_data_type, revision=model_revision
)

# Load VAE
vae = AutoencoderKL.from_pretrained(
    model_id, subfolder="vae", torch_dtype=model_data_type, revision=model_revision
)

print("Model components loaded successfully.")

# Step 2: Initialize Smash Config
print("Initializing Smash Config...")
smash_config = SmashConfig()
smash_config['quantizers'] = ['quanto']
smash_config['quant_quanto_calibrate'] = False
smash_config['quant_quanto_weight_bits'] = 'qint8'  # Options: "qfloat8", "qint2", "qint4", "qint8"

# Step 3: Smash the Model
print("Smashing the model (this may take a few minutes)...")
transformer = smash(
    model=transformer,
    token=PRUNA_TOKEN,
    smash_config=smash_config,
)
text_encoder_2 = smash(
    model=text_encoder_2,
    token=PRUNA_TOKEN,
    smash_config=smash_config,
)

print("Model smashing completed.")

# Step 4: Run the Model
print("Setting up the pipeline...")
pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=transformer
)

# Move components to GPU
# print("Moving model components to GPU...")
# pipe.text_encoder.to('cuda')
# pipe.vae.to('cuda')
# pipe.transformer.to('cuda')
# pipe.text_encoder_2.to('cuda')

# Enable additional memory savings (optional, but reduces speed)
vae.enable_tiling()
vae.enable_slicing()
# pipe.enable_sequential_cpu_offload()

# Run inference
print("Running inference...")
prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt=prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

# Save the output image
output_filename = "output_image.png"
image.save(output_filename)
print(f"Inference completed. Image saved as {output_filename}.")
