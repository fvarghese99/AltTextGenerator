# tests/test_image_utils.py

import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from PIL import Image
from io import BytesIO
from src.helpers.image_utils import find_images_without_alt_text, has_alt_text


@pytest.fixture
def setup_temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mix of image and non-image files
        image_files = [
            "image1.jpg",
            "image2.jpeg",
            "image3.png",
            "image4.gif",
            "image5.webp",
            "image6.svg",  # Add SVG file
            "document.txt",
            "script.py",
            "archive.zip",
        ]
        for file in image_files:
            open(os.path.join(tmpdir, file), 'a').close()
        yield tmpdir


def test_has_alt_text_true():
    with patch.object(Image, 'open') as mock_open:
        # Create the mock image
        mock_img = MagicMock()
        # If your code checks format, set it:
        mock_img.format = "JPEG"
        # Make getexif() return a dict with tag 270
        mock_img.getexif.return_value = {270: "Sample alt text"}
        # Ensure the context manager returns this mock image
        mock_open.return_value.__enter__.return_value = mock_img

        assert has_alt_text("fake_path.jpg") is True


def test_has_alt_text_false():
    # Mock an image with no EXIF data or empty description
    mock_exif = {270: ""}
    mock_img = MagicMock()
    mock_img.getexif.return_value = mock_exif

    with patch.object(Image, 'open', return_value=mock_img):
        assert has_alt_text("fake_path.jpg") is False


def test_has_alt_text_exception():
    # Force an exception when opening the image
    with patch.object(Image, 'open', side_effect=OSError("Cannot open")):
        assert has_alt_text("fake_path.jpg") is False


def test_has_alt_text_svg():
    # Test SVG file handling
    mock_img = MagicMock()
    mock_img.getexif.return_value = {270: "SVG alt text"}

    with patch('cairosvg.svg2png', return_value=b'fake_png_data') as mock_svg2png, \
            patch('PIL.Image.open') as mock_open:
        # Mock BytesIO to return our mock image
        mock_open.return_value.__enter__.return_value = mock_img

        result = has_alt_text("test.svg")

        # Check that svg2png was called for SVG conversion
        mock_svg2png.assert_called_once_with(url="test.svg")
        assert result is True


def test_has_alt_text_svg_without_alt():
    # Test SVG without alt text
    mock_img = MagicMock()
    mock_img.getexif.return_value = {}  # No alt text

    with patch('cairosvg.svg2png', return_value=b'fake_png_data') as mock_svg2png, \
            patch('PIL.Image.open') as mock_open:
        # Mock BytesIO to return our mock image
        mock_open.return_value.__enter__.return_value = mock_img

        result = has_alt_text("test.svg")

        # Check that svg2png was called for SVG conversion
        mock_svg2png.assert_called_once_with(url="test.svg")
        assert result is False


def test_has_alt_text_svg_conversion_error():
    # Test error during SVG conversion
    with patch('cairosvg.svg2png', side_effect=Exception("SVG conversion error")):
        assert has_alt_text("test.svg") is False


def test_find_images_without_alt_text(setup_temp_dir):
    tmpdir = setup_temp_dir

    # All files types should be included
    expected_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg')
    expected_images = []

    for root, _, files in os.walk(tmpdir):
        for file in files:
            if file.lower().endswith(expected_extensions):
                expected_images.append(os.path.join(root, file))

    # Mock has_alt_text to always return False
    with patch('src.helpers.image_utils.has_alt_text', return_value=False):
        found_images = find_images_without_alt_text(tmpdir)
        assert sorted(found_images) == sorted(expected_images)


def test_find_images_with_alt_text(setup_temp_dir):
    tmpdir = setup_temp_dir

    # Mock has_alt_text to always return True (all images have alt text)
    with patch('src.helpers.image_utils.has_alt_text', return_value=True):
        found_images = find_images_without_alt_text(tmpdir)
        assert found_images == []  # Should return empty list


def test_find_images_mixed_alt_text(setup_temp_dir):
    tmpdir = setup_temp_dir

    # Files with these patterns will return False for has_alt_text
    patterns_without_alt = ['image1', 'image3', 'image5']

    def mock_has_alt_text(path):
        filename = os.path.basename(path)
        for pattern in patterns_without_alt:
            if pattern in filename:
                return False
        return True

    with patch('src.helpers.image_utils.has_alt_text', side_effect=mock_has_alt_text):
        found_images = find_images_without_alt_text(tmpdir)

        # Only images matching the patterns should be returned
        assert len(found_images) == len(patterns_without_alt)
        for img in found_images:
            assert any(pattern in os.path.basename(img) for pattern in patterns_without_alt)


def test_find_images_with_no_images():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create non-image files only
        non_image_files = ["file1.txt", "file2.md", "file3.py"]
        for file in non_image_files:
            open(os.path.join(tmpdir, file), 'a').close()
        found_images = find_images_without_alt_text(tmpdir)
        assert found_images == []


def test_find_images_nonexistent_directory():
    with pytest.raises(FileNotFoundError):
        find_images_without_alt_text("/path/to/nonexistent/directory")