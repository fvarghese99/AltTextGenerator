import os
import subprocess
import re
import shutil
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

#########################
# HELPER: Sanitise Filename
#########################
def sanitise_filename(text: str, max_length: int = 50) -> str:
    r"""
    1) Strip leading/trailing whitespace
    2) Remove disallowed characters (\/:*?"<>|)
    3) Split into words and add them in order
       - If a word can't fully fit, stop adding more
    4) Capitalise the first letter
    """
    # 1. Strip whitespace
    text = text.strip()

    # 2. Remove disallowed characters
    text = re.sub(r'[\\\/:*?"<>|]', '', text)

    # 3. Split into words and build a truncated string
    tokens = text.split()
    new_text = ""
    for token in tokens:
        if not new_text:
            # If this is the first word, add it if it fits
            if len(token) <= max_length:
                new_text = token
            else:
                break  # If even the first word is too long, skip altogether
        else:
            # Check if we can add "space + token" in full
            if len(new_text) + 1 + len(token) <= max_length:
                new_text += " " + token
            else:
                # Stop adding words once we can't fit one in full
                break

    # 4. Capitalise the first letter if not empty
    if new_text:
        new_text = new_text[0].upper() + new_text[1:]

    return new_text

#########################
# 1. Identify Images
#########################
def find_images_without_alt_text(folder_path):
    """
    For this example, we simply list all images in a folder.
    """
    supported_extensions = (".jpg", ".jpeg", ".png", ".gif", ".webp")
    image_paths = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(supported_extensions):
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

    # Dynamically get terminal width, fallback to 100
    try:
        import shutil
        width = shutil.get_terminal_size().columns
    except OSError:
        width = 100
    print("-" * width)

    # Move model to the selected device
    model = model.to(device)
    return processor, model, device

def generate_caption(processor, model, device, image_path):
    """
    Open any image, convert to RGB, optionally resize, then generate a basic caption.
    """
    # 1. Open the image
    image = Image.open(image_path).convert("RGB")

    # 2. Resize if needed (helps avoid shape issues on MPS)
    image = image.resize((512, 512))

    # 3. Prepare BLIP2 input
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 4. Generate the caption
    with torch.no_grad():
        outputs = model.generate(**inputs)

    # 5. Decode to text
    caption_list = processor.batch_decode(outputs, skip_special_tokens=True)
    return caption_list[0] if caption_list else ""

#########################
# 3. Store/Update Alt Text
#    + Rename Image
#########################
def update_alt_text(image_path, alt_text):
    """
    Example function to show how you might:
      - Print the alt text.
      - Rename the file using the alt text (sanitised).
    """
    print(f"Image Path: {image_path}")
    print(f"Generated Alt Text: {alt_text}")

    # ------ Rename Logic ------
    # 1. Get the original extension
    _, ext = os.path.splitext(image_path)

    # 2. Sanitise alt text to create a safe filename
    safe_alt_text = sanitise_filename(alt_text)

    # 3. Construct new file path
    dir_name = os.path.dirname(image_path)
    new_filename = f"{safe_alt_text}{ext}"  # e.g. 'my_alt_text.jpg'
    new_path = os.path.join(dir_name, new_filename)

    # 4. Rename the file if alt_text isn't empty
    if safe_alt_text:
        try:
            os.rename(image_path, new_path)
            print(f"Renamed image to: {new_path}")
        except OSError as e:
            print(f"Failed to rename file: {e}")
    else:
        print("No alt text to rename file.")

    # Dynamically get terminal width, fallback to 100
    try:
        width = shutil.get_terminal_size().columns
    except OSError:
        width = 100
    print("-" * width)

#########################
# MAIN SCRIPT
#########################
def main():
    # Prompt the user for an image folder
    folder_path = input("Enter the path to the folder with images (or press Enter for 'img'): ").strip()
    if not folder_path:
        folder_path = "img"  # default fallback

    # Step 1: Identify images needing alt text
    images = find_images_without_alt_text(folder_path)
    if not images:
        print("No images found without alt text in that folder.")
        return

    # Step 2: Load the BLIP2 model once, and choose the device automatically
    processor, model, device = load_blip2_model()

    # Step 3: Loop through images and handle
    for image_path in images:
        alt_text = generate_caption(processor, model, device, image_path)
        update_alt_text(image_path, alt_text)

if __name__ == "__main__":
    main()