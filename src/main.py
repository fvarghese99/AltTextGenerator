# src/main.py

import os
import platform
from .helpers.file_utils import sanitise_filename
from .helpers.image_utils import find_images_without_alt_text
from .models.blip2 import load_blip2_model, generate_caption


def update_alt_text(image_path, alt_text):
    """
    Print and rename image files based on generated alt text,
    ensuring that filenames are unique to prevent overwriting.
    """
    print(f"🖼️ Image Path: {image_path}")
    print(f"📝 Generated Alt Text: {alt_text}")

    _, ext = os.path.splitext(image_path)
    safe_alt_text = sanitise_filename(alt_text)
    dir_name = os.path.dirname(image_path)

    # If sanitised alt text is empty, skip renaming
    if not safe_alt_text:
        print("⚠️ No alt text to rename file.")
        # Print separator and return
        try:
            import shutil
            width = shutil.get_terminal_size().columns
        except OSError:
            width = 100
        width = min(width, 100)
        print("-" * width)
        return

    # Otherwise, proceed with the rename
    new_filename = f"{safe_alt_text}{ext}"
    new_path = os.path.join(dir_name, new_filename)

    # Check if the new filename already exists (and isn't the same as the original file)
    counter = 1
    base_name = safe_alt_text
    while os.path.exists(new_path) and os.path.abspath(new_path) != os.path.abspath(image_path):
        # Append a counter (or you could use a short UUID)
        new_filename = f"{base_name}_{counter}{ext}"
        new_path = os.path.join(dir_name, new_filename)
        counter += 1

    try:
        os.rename(image_path, new_path)
        print(f"✅ Renamed image to: {new_path}")
    except OSError as e:
        print(f"❌ Failed to rename file: {e}")

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
        print(f"❌ The folder '{folder_path}' does not exist.")
        return

    images = find_images_without_alt_text(folder_path)
    if not images:
        print("⚠️ No images found without alt text in that folder.")
        return

    print("\n📥 Loading BLIP-2 model (this may take a while)...")
    processor, model, device = load_blip2_model()

    # Initial processing: update images missing alt text
    for image_path in images:
        alt_text = generate_caption(processor, model, device, image_path)
        update_alt_text(image_path, alt_text)

    print("\n✅ Initial processing completed.")

    # Verification loop: check if filenames match their alt text
    print("\n🔍 Verifying that filenames match their generated alt text...")

    mismatched_images = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                full_path = os.path.join(root, file)
                # Regenerate alt text for verification
                alt_text = generate_caption(processor, model, device, full_path)
                expected_name = sanitise_filename(alt_text)
                current_name = os.path.splitext(file)[0]

                if expected_name != current_name:
                    mismatched_images.append(full_path)

    if mismatched_images:
        print("⚠️ The following images have filenames that do not match their generated alt text:")
        try:
            import shutil
            width = shutil.get_terminal_size().columns
        except OSError:
            width = 100
            width = min(width, 100)  # Limit width to 100
        print("-" * width)
        for image in mismatched_images:
            print(f" - {image}")
            # Regenerate the caption and update the image
            alt_text = generate_caption(processor, model, device, image)
            update_alt_text(image, alt_text)
    else:
        print("✅ All images have filenames matching their alt text.")

if __name__ == "__main__":
    main()