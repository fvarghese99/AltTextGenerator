# src/models/blip2.py

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def load_blip2_model():
    """
    Load the BLIP2 model and processor from Hugging Face.
    Automatically selects the appropriate device.
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
    return processor, model, device

def generate_caption(processor, model, device, image_path):
    """
    Open an image and generate a caption using the BLIP2 model.
    """
    if processor is None or model is None:
        raise AttributeError("Model or processor is None.")

    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs)

    caption_list = processor.batch_decode(outputs, skip_special_tokens=True)
    return caption_list[0] if caption_list else ""