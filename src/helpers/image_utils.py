# src/helpers/image_utils.py

import os
from PIL import Image
from typing import List

def has_alt_text(image_path: str) -> bool:
    """
    Check if the image contains alt text metadata.
    For JPEG images, it checks the EXIF 'ImageDescription' (tag 270).
    """
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
            if exif and 270 in exif and exif[270]:
                return True
    except Exception as e:
        print(f"Error checking alt text for {image_path}: {e}")
    return False

def find_images_without_alt_text(
    folder_path: str,
    supported_extensions: tuple = (".jpg", ".jpeg", ".png", ".gif", ".webp")
) -> List[str]:
    """
    Finds all images without alt text in a directory.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")

    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(supported_extensions):
                full_path = os.path.join(root, file)
                if not has_alt_text(full_path):
                    image_paths.append(full_path)
    return image_paths