# src/models/blip2.py

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from typing import Tuple, Any, List, Dict

def load_blip2_model() -> Tuple[Any, Any, torch.device]:
    """
    Load the BLIP2 model and processor from Hugging Face.
    Automatically selects the appropriate device and sets the model to eval() mode.
    """
    print("Loading BLIP2 model...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU).")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon).")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU detected).")

    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    return processor, model, device

def _process_image(image_path: str) -> Image.Image:
    """Process an image by opening it, converting to RGB, and resizing."""
    image = Image.open(image_path).convert("RGB")
    return image.resize((512, 512))

def _generate_captions_from_inputs(processor: Any, model: Any, device: torch.device, inputs: dict) -> list:
    """Generate captions from preprocessed inputs using the BLIP2 model."""
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return processor.batch_decode(outputs, skip_special_tokens=True)

def generate_caption(processor: Any, model: Any, device: torch.device, image_path: str) -> str:
    """Open an image and generate a caption using the BLIP2 model."""
    if processor is None or model is None:
        raise AttributeError("Model or processor is None.")

    try:
        image = _process_image(image_path)
        inputs = processor(images=image, return_tensors="pt")
        caption_list = _generate_captions_from_inputs(processor, model, device, inputs)
        return caption_list[0] if caption_list else ""
    except Exception as e:
        print(f"Error generating caption for {image_path}: {e}")
        return ""

def generate_captions(processor: Any, model: Any, device: torch.device, image_paths: List[str]) -> Dict[str, str]:
    """Generate captions for a batch of images."""
    images = []
    valid_paths = []
    for path in image_paths:
        try:
            img = _process_image(path)
            images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(f"Error processing image {path}: {e}")
    if not images:
        return {}
    try:
        inputs = processor(images=images, return_tensors="pt")
        caption_list = _generate_captions_from_inputs(processor, model, device, inputs)
        return dict(zip(valid_paths, caption_list))
    except Exception as e:
        print(f"Error during batch caption generation: {e}")
        return {}