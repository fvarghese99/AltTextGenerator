# src/helpers/image_utils.py

import os

def find_images_without_alt_text(folder_path):
    """
    Finds all images without alt text in a directory.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")

    supported_extensions = (".jpg", ".jpeg", ".png", ".gif", ".webp")
    image_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files
        if file.lower().endswith(supported_extensions)
    ]
    return image_paths