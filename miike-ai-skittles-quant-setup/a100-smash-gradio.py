import os
import torch
import gradio as gr
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
MODEL_ID = "miike-ai/skittles-v2"
MODEL_REVISION = "main"
TEXT_MODEL_ID = "openai/clip-vit-large-patch14"
MODEL_DATA_TYPE = torch.bfloat16

# Global pipeline variable
PIPELINE = None

def load_and_quantize_model():
    """Loads and quantizes the model components, returning the pipeline."""
    print("Loading model components...")

    # Load CLIP tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(TEXT_MODEL_ID, torch_dtype=MODEL_DATA_TYPE)
    text_encoder = CLIPTextModel.from_pretrained(TEXT_MODEL_ID, torch_dtype=MODEL_DATA_TYPE)

    # Load T5 tokenizer and text encoder
    tokenizer_2 = T5TokenizerFast.from_pretrained(
        MODEL_ID, subfolder="tokenizer_2", torch_dtype=MODEL_DATA_TYPE, revision=MODEL_REVISION
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder_2", torch_dtype=MODEL_DATA_TYPE, revision=MODEL_REVISION
    )

    # Load scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL_ID, subfolder="scheduler", revision=MODEL_REVISION
    )

    # Load transformer
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=MODEL_DATA_TYPE, revision=MODEL_REVISION
    )

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=MODEL_DATA_TYPE, revision=MODEL_REVISION
    )

    print("Model components loaded successfully.")

    # Initialize Smash Config
    print("Initializing Smash Config...")
    smash_config = SmashConfig()
    smash_config['quantizers'] = ['quanto']
    smash_config['quant_quanto_calibrate'] = False
    smash_config['quant_quanto_weight_bits'] = 'qint4'  # Options: "qfloat8", "qint2", "qint4", "qint8"

    # Smash the Model
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

    # Create pipeline
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
    print("Moving model components to GPU...")
    pipe.text_encoder.to('cuda')
    pipe.vae.to('cuda')
    pipe.transformer.to('cuda')
    pipe.text_encoder_2.to('cuda')

    # Enable additional memory savings (optional, but reduces speed)
    vae.enable_tiling()
    vae.enable_slicing()
    pipe.enable_sequential_cpu_offload()

    print("Pipeline setup complete.")
    return pipe

# Load the pipeline on startup
print("Initializing model...")
PIPELINE = load_and_quantize_model()

def generate_image(prompt, guidance_scale=0.0, num_inference_steps=50):
    """Generates an image based on the provided prompt."""
    print(f"Generating image for prompt: {prompt}")
    image = PIPELINE(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    return image

# Gradio Interface
interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Describe the image you want to generate."),
        gr.Slider(0, 10, value=0.0, step=0.1, label="Guidance Scale"),
        gr.Slider(1, 100, value=50, step=1, label="Number of Inference Steps")
    ],
    outputs=gr.Image(label="Generated Image"),
    title="Flux Model Image Generator",
    description="Generate images using the preloaded and quantized Flux model with Skittles-v2.",
)

if __name__ == "__main__":
    interface.launch(share=True)
