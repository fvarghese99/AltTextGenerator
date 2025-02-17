# tests/test_image_utils.py

import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from PIL import Image
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

def test_find_images_without_alt_text(setup_temp_dir):
    tmpdir = setup_temp_dir
    expected_images = [
        os.path.join(tmpdir, "image1.jpg"),
        os.path.join(tmpdir, "image2.jpeg"),
        os.path.join(tmpdir, "image3.png"),
        os.path.join(tmpdir, "image4.gif"),
        os.path.join(tmpdir, "image5.webp"),
    ]
    found_images = find_images_without_alt_text(tmpdir)
    assert sorted(found_images) == sorted(expected_images)

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