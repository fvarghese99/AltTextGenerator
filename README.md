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
- pytest 7.4.4 or above for testing

## Installation

1. Clone the repository or download the script:
   ```bash
   git clone https://github.com/fvarghese99/AltTextGenerator.git
   ```
    Navigate to the project directory:
    ```bash
   cd AltTextGenerator
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

## Usage

1. Place the images in the default img folder or create your own folder.
2. Run the script:
    ```bash
    python main.py
    ```
3. When prompted, enter the path to the folder containing images or press Enter to use the default img folder:
    ```bash
    Enter the path to the folder with images (or press Enter for 'img'): 
    ```
4. The script will:
    - Generate alt text for each image.
    - Rename the images based on the generated alt text.
5. View the renamed images in the specified folder.

## Notes

- Images are resized to 512x512 pixels for compatibility with the BLIP2 model.
- Ensure that the folder path provided contains supported image formats.

## Example Output

```
Enter the path to the folder with images (or press Enter for 'img'): 
Loading BLIP2 model...
Using MPS (Apple Silicon).
------------------------------------------------------------------
Image Path: img/a brick building with a sign on the side.jpg
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