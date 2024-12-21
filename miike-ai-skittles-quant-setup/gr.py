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

# Load the pipeline
def load_pipeline(model_id, torch_dtype, qtype=None, device="cpu"):
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, use_safetensors=True).to(device)

    if qtype:
        quantize(pipe.transformer, weights=qtype)
        freeze(pipe.transformer)
        quantize(pipe.text_encoder, weights=qtype)
        freeze(pipe.text_encoder)

    pipe.set_progress_bar_config(disable=True)
    return pipe

# Get device memory
def get_device_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated()
    elif device.type == "mps":
        torch.mps.empty_cache()
        return torch.mps.current_allocated_memory()
    return None

# Gradio interface function
def generate_image(model_id, prompt, torch_dtype, qtype, device):
    start_time = time.time()

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    pipeline = load_pipeline(
        model_id, TORCH_DTYPES[torch_dtype], QTYPES[qtype] if qtype != "none" else None, device
    )

    memory = get_device_memory(device)
    memory_info = (
        f"{device.type} device memory: {memory / 2**30:.2f} GB."
        if memory is not None
        else "Memory information not available."
    )

    if qtype == "int4" and device.type == "cuda":
        raise ValueError("This example does not work (yet) for int4 on CUDA")

    image = pipeline(
        prompt=prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_images_per_prompt=1,
        generator=torch.manual_seed(0),
    ).images[0]

    elapsed_time = time.time() - start_time
    return image, f"Image generated in {elapsed_time:.2f} seconds.\n{memory_info}"

# Gradio Interface
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(value="miike-ai/skittles-v2", label="Model ID"),
        gr.Textbox(value="ghibli style, a fantasy landscape with castles", label="Prompt"),
        gr.Radio(choices=list(TORCH_DTYPES.keys()), value="bf16", label="Torch DType"),
        gr.Radio(choices=list(QTYPES.keys()), value="int4", label="Quantization Type (QType)"),
        gr.Textbox(value=None, label="Device (leave blank for auto-detect)")
    ],
    outputs=[
        gr.Image(label="Generated Image"),
        gr.Textbox(label="Performance Details"),
    ],
    title="Diffusion Image Generator with Quantization",
    description="Generate images using a diffusion pipeline with options for Torch DType and quantization.",
)

if __name__ == "__main__":
    iface.launch(share=True)
