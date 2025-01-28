# src/main.py

import os
import platform
from .helpers.file_utils import sanitise_filename
from .helpers.image_utils import find_images_without_alt_text
from .models.blip2 import load_blip2_model, generate_caption

def update_alt_text(image_path, alt_text):
    """
    Print and rename image files based on generated alt text.
    """
    print(f"Image Path: {image_path}")
    print(f"Generated Alt Text: {alt_text}")

    _, ext = os.path.splitext(image_path)
    safe_alt_text = sanitise_filename(alt_text)
    dir_name = os.path.dirname(image_path)
    new_filename = f"{safe_alt_text}{ext}"
    new_path = os.path.join(dir_name, new_filename)

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
        import shutil
        width = shutil.get_terminal_size().columns
    except OSError:
        width = 100
        width = min(width, 100)  # Limit width to 100
    print("-" * width)

def main():
    # Detect operating system and set default folder path
    system_os = platform.system()
    if system_os == "Darwin":  # macOS
        default_folder = "./data/img"
    elif system_os == "Windows":
        default_folder = "../data/img"
    else:  # Fallback for other OS like Linux
        default_folder = "./data/img"

    folder_path = input(f"Enter the path to the folder with images (or press Enter for '{default_folder}'): ").strip()
    if not folder_path:
        folder_path = default_folder

    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    images = find_images_without_alt_text(folder_path)
    if not images:
        print("No images found without alt text in that folder.")
        return

    processor, model, device = load_blip2_model()

    for image_path in images:
        alt_text = generate_caption(processor, model, device, image_path)
        update_alt_text(image_path, alt_text)

if __name__ == "__main__":
    main()