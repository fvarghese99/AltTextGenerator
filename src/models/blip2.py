# src/models/blip2.py

import os
import torch
import gc
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from typing import Tuple, Any, List, Dict
import cairosvg
from io import BytesIO
import time


def load_blip2_model() -> Tuple[Any, Any, torch.device]:
    """
    Load the BLIP2 model and processor from Hugging Face.
    Automatically selects the appropriate device and sets the model to eval() mode.
    Includes optimizations for CUDA devices.
    """
    print("Loading BLIP2 model...")

    # Clear CUDA cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Load processor first
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

    # Determine device before loading model to memory
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU).")

        # Get CUDA properties
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name} with {props.total_memory / 1024 ** 3:.1f} GB memory")

        # Load model with half precision to save memory on CUDA
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon).")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU detected).")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

    # Move model to device
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    return processor, model, device


def _preprocess_image_batch(image_paths: List[str]) -> List[Tuple[str, Image.Image]]:
    """
    Preprocess a batch of images in parallel using multiprocessing.
    Returns a list of valid (path, image) tuples.
    """
    valid_images = []
    for path in image_paths:
        try:
            # Get file extension
            ext = os.path.splitext(path)[1].lower()

            if ext == '.svg':
                # Convert SVG to PNG in memory
                try:
                    png_data = cairosvg.svg2png(url=path)
                    image = Image.open(BytesIO(png_data))
                except Exception as e:
                    print(f"Error converting SVG file {path}: {e}")
                    continue
            else:
                # Open regular image file
                image = Image.open(path)

            # Convert to RGB (handles RGBA, palette images, etc.)
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')

            # Resize for BLIP-2
            resized_img = image.resize((512, 512))
            valid_images.append((path, resized_img))
        except Exception as e:
            print(f"Error processing image {path}: {e}")

    return valid_images


def generate_captions(processor: Any, model: Any, device: torch.device, image_paths: List[str],
                      batch_size: int = None) -> Dict[str, str]:
    """
    Generate captions for a list of images using optimal batching based on device type.

    Args:
        processor: BLIP2 processor
        model: BLIP2 model
        device: Device to use (cuda, mps, cpu)
        image_paths: List of paths to image files
        batch_size: Override batch size (if None, will be calculated automatically)

    Returns:
        Dictionary mapping image paths to generated captions
    """
    if not image_paths:
        return {}

    # Calculate optimal batch size if not provided
    if batch_size is None:
        if device.type == 'cuda':
            # Get available CUDA memory and calculate safe batch size
            # This is a conservative approach for Windows
            try:
                free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                free_gb = free_mem / (1024 ** 3)
                # Estimate: each image takes roughly 40-50MB in GPU memory
                batch_size = max(1, min(4, int(free_gb / 0.25)))
            except (torch.cuda.CUDAError, RuntimeError, AttributeError) as e:
                print(f"CUDA memory calculation failed: {e}")
                # Fallback to a safe value
                batch_size = 2
        elif device.type == 'mps':
            batch_size = 2  # Conservative for MPS
        else:  # CPU
            batch_size = 1  # CPU processes one at a time to avoid memory issues

    print(f"Using batch size of {batch_size} for {device.type} device")

    # Process all images and keep only valid ones
    start_time = time.time()
    print(f"Preprocessing {len(image_paths)} images...")
    valid_images = _preprocess_image_batch(image_paths)
    print(
        f"Successfully preprocessed {len(valid_images)} of {len(image_paths)} images in {time.time() - start_time:.2f}s")

    if not valid_images:
        return {}

    # Process images in batches
    results = {}
    num_batches = (len(valid_images) + batch_size - 1) // batch_size

    for i in range(0, len(valid_images), batch_size):
        batch = valid_images[i:i + batch_size]
        paths = [path for path, _ in batch]
        images = [img for _, img in batch]

        batch_start = time.time()
        print(f"Processing batch {i // batch_size + 1}/{num_batches} ({len(batch)} images)...")

        try:
            # Pack all images into a single tensor batch
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Clear unnecessary CPU memory
            del images

            with torch.no_grad():
                if device.type == "cuda":
                    # Use mixed precision to speed up inference on NVIDIA GPUs
                    with torch.amp.autocast('cuda'):
                        outputs = model.generate(**inputs)
                else:
                    outputs = model.generate(**inputs)

            captions = processor.batch_decode(outputs, skip_special_tokens=True)

            # Map captions back to their paths
            batch_results = dict(zip(paths, captions))
            results.update(batch_results)

            # Clear CUDA cache after each batch
            if device.type == "cuda":
                del inputs, outputs
                torch.cuda.empty_cache()
                gc.collect()

            batch_time = time.time() - batch_start
            print(
                f"Batch {i // batch_size + 1}/{num_batches} completed in {batch_time:.2f}s ({batch_time / len(batch):.2f}s per image)")

        except Exception as e:
            print(f"Error during batch caption generation: {e}")
            # Try to process each image individually if batch processing failed
            print("Falling back to individual processing...")
            for path, img in batch:
                try:
                    inputs = processor(images=img, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        if device.type == "cuda":
                            with torch.amp.autocast('cuda'):
                                outputs = model.generate(**inputs)
                        else:
                            outputs = model.generate(**inputs)

                    captions = processor.batch_decode(outputs, skip_special_tokens=True)
                    results[path] = captions[0]

                    # Clear CUDA cache
                    if device.type == "cuda":
                        del inputs, outputs
                        torch.cuda.empty_cache()
                except Exception as inner_e:
                    print(f"Error with individual image {path}: {inner_e}")

    # Final memory cleanup
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    return results


def generate_caption(processor: Any, model: Any, device: torch.device, image_path: str) -> str:
    """
    Generate a caption for a single image (convenience function).
    """
    results = generate_captions(processor, model, device, [image_path], batch_size=1)
    return results.get(image_path, "")