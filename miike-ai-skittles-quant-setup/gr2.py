import gradio as gr
import gc
import time
import torch
from diffusers import DiffusionPipeline
from optimum.quanto import freeze, qfloat8, qint4, qint8, quantize

# Constants
NUM_INFERENCE_STEPS = 28
TORCH_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
QTYPES = {
    "fp8": qfloat8,
    "int8": qint8,
    "int4": qint4,
    "none": None,
}

# Global variables for the model pipeline
PIPELINE = None
MEMORY_INFO = None


def load_pipeline_on_startup(model_id, torch_dtype, qtype, device):
    """Load the pipeline during startup."""
    global PIPELINE, MEMORY_INFO
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    PIPELINE = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, use_safetensors=True).to(device)

    if qtype != "none":
        quantize(PIPELINE.transformer, weights=qtype)
        freeze(PIPELINE.transformer)
        quantize(PIPELINE.text_encoder, weights=qtype)
        freeze(PIPELINE.text_encoder)

    PIPELINE.set_progress_bar_config(disable=True)
    MEMORY_INFO = get_device_memory(device)
    return f"Pipeline loaded successfully on {device}."


def get_device_memory(device):
    """Check and return memory usage of the device."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        return f"{torch.cuda.memory_allocated() / 2**30:.2f} GB allocated."
    elif device.type == "mps":
        torch.mps.empty_cache()
        return f"{torch.mps.current_allocated_memory() / 2**30:.2f} GB allocated."
    return "Memory info not available for this device."


def generate_image(prompt):
    """Generate an image using the preloaded pipeline."""
    if PIPELINE is None:
        raise RuntimeError("Pipeline is not loaded. Please check the setup.")

    start_time = time.time()
    image = PIPELINE(
        prompt=prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_images_per_prompt=1,
        generator=torch.manual_seed(0),
    ).images[0]
    elapsed_time = time.time() - start_time
    return image, f"Image generated in {elapsed_time:.2f} seconds.\n{MEMORY_INFO}"


# Gradio Interface
iface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(value="ghibli style, a fantasy landscape with castles", label="Prompt"),
    outputs=[
        gr.Image(label="Generated Image"),
        gr.Textbox(label="Performance Details"),
    ],
    title="Preloaded Diffusion Image Generator",
    description="Generate images using a preloaded diffusion pipeline with quantization.",
)


if __name__ == "__main__":
    # Load the pipeline during startup
    print("Loading pipeline on startup...")
    model_id = "miike-ai/skittles-v2"
    torch_dtype = "fp16"
    qtype = "int8"
    device = None  # Auto-detect device
    load_message = load_pipeline_on_startup(model_id, TORCH_DTYPES[torch_dtype], QTYPES[qtype], device)
    print(load_message)
    iface.launch(share=True)
