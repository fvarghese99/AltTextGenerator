import os
import subprocess
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration


#########################
# 1. Identify Images
#########################
def find_images_without_alt_text(folder_path):
    """
    For this example, we simply list all images in a folder.
    In a real scenario, you would query your CMS (e.g. Umbraco)
    to find media items missing alt text.
    """
    supported_extensions = (".jpg", ".jpeg", ".png", ".gif", ".webp")
    image_paths = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(supported_extensions):
                # In real usage, check if your CMS has an alt text stored.
                # If not, add to list. Here, we assume all images lack alt text.
                image_paths.append(os.path.join(root, file))

    return image_paths


#########################
# 2. Generate Captions
#########################
def load_blip2_model():
    """
    Load the BLIP2 model and processor from Hugging Face.

    We automatically detect which device to run on:
      - 'cuda' if an NVIDIA GPU is available (Windows/Linux typically),
      - 'mps' if on Apple Silicon (M1/M2),
      - otherwise 'cpu'.
    """
    print("Loading BLIP2 model...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-base")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-base")

    # Automatic device selection
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
    Use the BLIP2 model to produce a basic caption describing the image.
    """
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    # Move inputs to the selected device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    caption = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    return caption


#########################
# 3. Refine with Ollama
#########################
def refine_caption_with_ollama(caption):
    """
    Uses the Ollama command-line interface to refine the caption.

    Make sure 'ollama' is in your PATH or provide the full path to the ollama binary.
    """
    prompt = (
        f"Refine the following image description into a concise alt text: {caption}\n"
        "Keep it brief and descriptive."
    )

    # Note: on Windows, you might need to use shell=True or other adjustments.
    # Also consider using the subprocess 'text' or 'capture_output' arguments if needed.
    result = subprocess.run(
        ["ollama", "prompt", prompt],
        stdout=subprocess.PIPE,
        universal_newlines=True
    )

    refined_text = result.stdout.strip()
    return refined_text


#########################
# 4. Store/Update Alt Text
#########################
def update_alt_text_in_cms(image_path, alt_text):
    """
    Dummy function to show how you might save alt text.
    In practice, you’d interact with Umbraco’s API or database.
    """
    # Example: print or log the result
    # In real usage, call your Umbraco endpoint to update alt text for the media item.
    print(f"Image Path: {image_path}")
    print(f"Generated Alt Text: {alt_text}")
    print("-" * 60)


#########################
# MAIN SCRIPT
#########################
def main():
    # Folder of images lacking alt text
    script_dir = os.path.dirname(__file__)
    folder_path = os.path.join(script_dir, "img")

    # Step 1: Identify images needing alt text
    images = find_images_without_alt_text(folder_path)
    if not images:
        print("No images found without alt text.")
        return

    # Step 2: Load the BLIP2 model once, and choose the device automatically
    processor, model, device = load_blip2_model()

    # Step 3: Loop through images and handle
    for image_path in images:
        print(f"Processing: {image_path}")

        # Generate initial caption
        initial_caption = generate_caption(processor, model, device, image_path)
        print(f"Initial Caption: {initial_caption}")

        # Refine with Ollama
        alt_text = refine_caption_with_ollama(initial_caption)

        # Step 4: Update alt text in your CMS
        update_alt_text_in_cms(image_path, alt_text)


if __name__ == "__main__":
    main()