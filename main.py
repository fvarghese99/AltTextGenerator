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
                # In a real system, check if your CMS has an alt text stored.
                # If not, add to list. Here, we assume all images lack alt text.
                image_paths.append(os.path.join(root, file))

    return image_paths

#########################
# 2. Generate Captions
#########################
def load_blip2_model():
    """
    Load the BLIP2 model and processor from Hugging Face.

    Automatically detects which device to run on:
     - 'cuda' if an NVIDIA GPU is available (Windows/Linux),
     - 'mps' if on Apple Silicon (M1/M2),
     - otherwise 'cpu'.
    """
    print("Loading BLIP2 model...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

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

    # Move model to the selected device
    model = model.to(device)
    return processor, model, device

def generate_caption(processor, model, device, image_path):
    """
    Open any image, convert to RGB, optionally resize, then generate a basic caption.
    """
    # 1. Open the image
    image = Image.open(image_path)

    # 2. Convert to RGB mode
    image = image.convert("RGB")

    # 3. Resize to avoid shape mismatches on MPS (and standardise input size)
    image = image.resize((512, 512))

    # 4. Prepare BLIP2 input
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 5. Generate the caption
    with torch.no_grad():
        outputs = model.generate(**inputs)

    caption = processor.batch_decode(outputs, skip_special_tokens=True)
    return caption

#########################
# 3. Refine with Ollama
#########################
def refine_caption_with_ollama(caption):
    """
    Uses Ollama to refine the caption. Your Ollama version expects a positional
    argument for the prompt (rather than stdin or --prompt).
    """
    prompt_text = (
        f"Refine the following image description into a concise alt text: {caption}\n"
        "Keep it brief and descriptive."
    )

    # Provide the prompt as a positional argument:
    result = subprocess.run(
        ["ollama", "run", prompt_text],
        stdout=subprocess.PIPE,
        text=True
    )
    return result.stdout.strip()

#########################
# 4. Store/Update Alt Text
#########################
def update_alt_text_in_cms(image_path, alt_text):
    """
    Dummy function to show how you might save alt text.
    In practice, you'd interact with Umbraco's API or database.
    """
    print(f"Image Path: {image_path}")
    print(f"Generated Alt Text: {alt_text}")
    print("-" * 80)

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

        # Generate initial caption (BLIP2)
        initial_caption = generate_caption(processor, model, device, image_path)
        print(f"Initial Caption: {initial_caption}")

        # Refine with Ollama
        alt_text = refine_caption_with_ollama(initial_caption)

        # Step 4: Update alt text in your "CMS"
        update_alt_text_in_cms(image_path, alt_text)

if __name__ == "__main__":
    main()