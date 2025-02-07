# AltTextGenerator

AltTextGenerator is a Python script designed to automatically generate alt text for images using the BLIP2 model. It also renames the images based on the generated alt text, ensuring accessible and SEO-friendly filenames.

## Features

- Automatically generates alt text for images using the BLIP2 model.
- Renames image files based on the generated alt text.
- Supports a variety of image formats, including `.jpg`, `.jpeg`, `.png`, `.gif`, and `.webp`.
- Automatically detects the device for processing (CUDA, MPS, or CPU).
- User-friendly interface for selecting the folder containing images.

## Requirements

- Python 3.12 or above
- Pillow (>=11.1.0,<12.0.0) for image handling
- Hugging Face Transformers (>=4.48.0,<5.0.0) library
- torch (>=2.5.1,<3.0.0) library

## Installation

1. Clone the repository or download the script:
   ```bash
   git clone https://github.com/fvarghese99/AltTextGenerator.git
   ```
    Navigate to the project directory:
    ```bash
   cd /path/to/AltTextGenerator
    ```
2. Create a virtual environment with custom name 'AltTextGeneratorEnv' (recommended):
    ```bash
    python -m venv AltTextGeneratorEnv
   ```
    Activate the virtual environment:   <br>
    For macOS/Linux <br>
   ```bash
    source AltTextGeneratorEnv/bin/activate  
   ```
    For Windows <br>
    ```bash
    AltTextGeneratorEnv\Scripts\activate     
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
   ```

## PyTorch Dependencies

To ensure you install the correct versions of torch, torchvision, and torchaudio that are compatible with your system (e.g. with CUDA support or CPU-only), please refer to the official [PyTorch Get Started Locally](https://pytorch.org/get-started/locally/) page.
<br><br>For example, if you are using pip and wish to install a CUDA-enabled version (replace cu126 with the CUDA version suited to your system), you might run:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## Usage

1. Place the images in the default img folder or create your own folder.
2. Run the script:
    ```bash
    python -m src.main
    ```
3. When prompted, enter the path to the folder containing images or press Enter to use the default img folder:
    ```bash
    Enter the path to the folder with images (or press Enter for './data/img'): 
    ```
4. The script will:
    - Generate alt text for each image.
    - Rename the images based on the generated alt text.
5. View the renamed images in the specified folder.

## Testing

To ensure the script is working correctly, follow these steps to run the tests:
1. Install the pytest package:
    ```bash
    pip install pytest
    ```
2. Run all tests:
    ```bash
    pytest tests/
    ```
3. Test details:
   - The tests include validation for:
     - The BLIP2 model functionality.
     - Utility functions for file and image processing.
     - End-to-end integration of image processing and renaming.<br>
  
4. Check the output to conform that all tests have passed. Example:
    ```bash
    ============================= test session starts ==============================
    platform darwin -- Python 3.12.8, pytest-7.4.4, pluggy-1.5.0
    rootdir: /AltTextGenerator
    plugins: cov-6.0.0
    collected 24 items

    tests/test_blip2.py ......                                              [ 25%]
    tests/test_file_utils.py ......                                         [ 50%]
    tests/test_image_utils.py ......                                        [ 75%]
    tests/test_main.py ......                                               [100%]

    ============================== 24 passed in 5.06s ===============================
    ```

## Notes

- Images are resized to 512x512 pixels for compatibility with the BLIP2 model.
- Ensure that the folder path provided contains supported image formats.

## Example Output

```
Enter the path to the folder with images (or press Enter for 'img'): 
Loading BLIP2 model...
Using MPS (Apple Silicon).
------------------------------------------------------------------
Image Path: img/img-1.jpg
Generated Alt Text: A brick building with a sign on the side

Renamed image to: img/A brick building with a sign on the side.jpg
------------------------------------------------------------------
```

## Supported Image Formats

- `.jpg`
- `.jpeg`
- `.png`
- `.gif`
- `.webp`

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org)
- [BLIP2](https://huggingface.co/Salesforce/blip2-opt-2.7b)