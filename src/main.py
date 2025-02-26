# src/main.py

import os
import platform
import time
from .helpers.file_utils import sanitise_filename
from .helpers.image_utils import find_images_without_alt_text
from .models.blip2 import load_blip2_model, generate_captions


def _get_terminal_width() -> int:
    """
    Get the width of the terminal for formatting output.
    """
    try:
        import shutil
        width = shutil.get_terminal_size().columns
        return min(width, 100)  # Limit width to 100
    except (OSError, ImportError):
        return 100


def update_alt_text(image_path: str, alt_text: str) -> None:
    """
    Print the image path and generated alt text, and rename the image file.
    """
    print(f"🖼️ Image Path: {image_path}")
    print(f"📝 Generated Alt Text: {alt_text}")

    # Extract the file extension
    _, ext = os.path.splitext(image_path)
    # Sanitise the alt text to create a safe filename
    safe_alt_text = sanitise_filename(alt_text)
    dir_name = os.path.dirname(image_path)

    # Skip renaming if the sanitised alt text is empty
    if not safe_alt_text or safe_alt_text.isspace():
        print("⚠️ No alt text to rename file.")
        print("-" * _get_terminal_width())
        return

    # Create a new filename using the sanitised alt text
    new_filename = f"{safe_alt_text}{ext}"
    new_path = os.path.join(dir_name, new_filename)

    # Handle duplicate filenames
    counter = 1
    base_name = safe_alt_text
    while os.path.exists(new_path) and os.path.abspath(new_path) != os.path.abspath(image_path):
        new_filename = f"{base_name}_{counter}{ext}"
        new_path = os.path.join(dir_name, new_filename)
        counter += 1

    try:
        # Rename the file
        os.rename(image_path, new_path)
        print(f"✅ Renamed image to: {new_path}")
    except (OSError, PermissionError) as e:
        print(f"❌ Failed to rename file: {e}")

    # Print a separator line
    print("-" * _get_terminal_width())


def batch_update_alt_text(caption_results: dict) -> None:
    """
    Update alt text for a batch of images.
    """
    for image_path, alt_text in caption_results.items():
        update_alt_text(image_path, alt_text)


def main():
    """
    Main function that drives the program with optimized batch processing.
    """
    # Record start time for performance measurement
    start_time = time.time()

    # Detect OS and set default folder path
    system_os = platform.system()
    if system_os == "Darwin":  # macOS
        default_folder = "./data/img"
    elif system_os == "Windows":
        default_folder = "../data/img"
    else:  # Linux or other
        default_folder = "./data/img"

    # Get folder path from user
    folder_path = input(f"Enter the path to the folder with images (or press Enter for '{default_folder}'): ").strip()
    if not folder_path:
        folder_path = default_folder

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"❌ The folder '{folder_path}' does not exist.")
        return

    # Find images without alt text
    print("\n🔍 Finding images without alt text...")
    images = find_images_without_alt_text(folder_path)
    if not images:
        print("⚠️ No images found without alt text in that folder.")
        return

    print(f"Found {len(images)} images without alt text.")

    # Load BLIP-2 model
    print("\n📥 Loading BLIP-2 model (this may take a while)...")
    processor, model, device = load_blip2_model()

    # Process initial images
    process_start = time.time()
    print(f"\n⏳ Generating alt text for {len(images)} images...")

    # Let the generate_captions function determine the best batch size
    captions = generate_captions(processor, model, device, images)

    print(f"✅ Generated captions for {len(captions)} images in {time.time() - process_start:.2f}s")

    # Update filenames based on alt text
    print("\n🔄 Updating filenames...")
    batch_update_alt_text(captions)

    print("\n✅ Initial processing completed.")

    # Verification step
    print("\n🔍 Verifying that filenames match their generated alt text...")
    verification_images = []

    # Find all image files in the folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.jfif', '.svg', '.webp')):
                full_path = os.path.join(root, file)

                # Skip images we just processed and renamed
                if full_path in captions:
                    # This image was just processed and renamed correctly
                    continue

                # Add to verification list
                verification_images.append(full_path)

    if verification_images:
        print(f"Found {len(verification_images)} images that need verification.")

        # Generate captions for verification images
        verification_start = time.time()
        verification_captions = generate_captions(processor, model, device, verification_images)
        print(f"✅ Generated verification captions in {time.time() - verification_start:.2f}s")

        # Check which images need renaming
        mismatched_images = {}
        for image_path, alt_text in verification_captions.items():
            expected_name = sanitise_filename(alt_text)
            current_name = os.path.splitext(os.path.basename(image_path))[0]

            if expected_name != current_name:
                mismatched_images[image_path] = alt_text

        # Update mismatched images
        if mismatched_images:
            print(f"⚠️ Found {len(mismatched_images)} images with mismatched filenames.")
            batch_update_alt_text(mismatched_images)
        else:
            print("✅ All verification images have correct filenames.")
    else:
        print("✅ No additional images need verification.")

    # Print final stats
    total_time = time.time() - start_time
    print(f"\n🎉 All done! Processed {len(images) + len(verification_images)} images in {total_time:.2f}s")

    # Device-specific performance message
    if device.type == "cuda":
        print("\n💡 CUDA Performance Note: If you experienced slow processing, try these tips:")
        print("  - Close other CUDA applications")
        print("  - Update your NVIDIA drivers")
        print("  - Try a smaller batch size (edit calculate_batch_size in models/blip2.py)")
        print("  - For larger image sets, consider using CPU mode if CUDA is unstable")


if __name__ == "__main__":
    main()