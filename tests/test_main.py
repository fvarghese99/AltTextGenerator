# tests/test_main.py

import pytest
from unittest.mock import patch, MagicMock
from src.main import update_alt_text, main

import torch
import os


@pytest.fixture
def mock_find_images():
    with patch('src.main.find_images_without_alt_text') as mock_func:
        yield mock_func


@pytest.fixture
def mock_load_blip2_model():
    with patch('src.main.load_blip2_model') as mock_func:
        yield mock_func


@pytest.fixture
def mock_generate_caption():
    with patch('src.main.generate_caption') as mock_func:
        yield mock_func


@pytest.fixture
def mock_update_alt_text():
    with patch('src.main.update_alt_text') as mock_func:
        yield mock_func


def test_sanitise_filename_valid(tmpdir):
    # Reuse the sanitise_filename function test
    pass  # Assuming already tested in test_file_utils.py


@patch('shutil.get_terminal_size', return_value=os.terminal_size((100, 24)))  # Mock terminal width to 100
def test_update_alt_text_success(monkeypatch, tmpdir):
    image_path = os.path.join(tmpdir, "image.jpg")
    with open(image_path, 'w') as f:
        f.write("fake image content")

    alt_text = "New Alt Text"
    new_filename = "New_alt_text.jpg"
    new_path = os.path.join(tmpdir, new_filename)

    with patch('src.main.sanitise_filename', return_value="New_alt_text") as mock_sanitise, \
            patch('os.rename') as mock_rename, \
            patch('builtins.print') as mock_print:
        update_alt_text(image_path, alt_text)

        mock_sanitise.assert_called_once_with(alt_text)
        mock_rename.assert_called_once_with(image_path, new_path)
        mock_print.assert_any_call(f"🖼️ Image Path: {image_path}")
        mock_print.assert_any_call(f"📝 Generated Alt Text: {alt_text}")
        mock_print.assert_any_call(f"✅ Renamed image to: {new_path}")
        mock_print.assert_any_call("-" * 100)


@patch('shutil.get_terminal_size', return_value=os.terminal_size((100, 24)))  # Mock terminal width to 100
def test_update_alt_text_no_alt_text(monkeypatch, tmpdir):
    image_path = os.path.join(tmpdir, "image.jpg")
    with open(image_path, 'w') as f:
        f.write("fake image content")

    alt_text = "   "  # After sanitisation, this should be empty

    with patch('src.main.sanitise_filename', return_value=""), \
            patch('os.rename') as mock_rename, \
            patch('builtins.print') as mock_print:
        update_alt_text(image_path, alt_text)

        mock_rename.assert_not_called()
        mock_print.assert_any_call("🖼️ Image Path: {}".format(image_path))
        mock_print.assert_any_call("📝 Generated Alt Text: {}".format(alt_text))
        mock_print.assert_any_call("⚠️ No alt text to rename file.")
        mock_print.assert_any_call("-" * 100)


@patch('shutil.get_terminal_size', return_value=os.terminal_size((100, 24)))  # Mock terminal width to 100
def test_update_alt_text_rename_failure(monkeypatch, tmpdir):
    image_path = os.path.join(tmpdir, "image.jpg")
    with open(image_path, 'w') as f:
        f.write("fake image content")

    alt_text = "ValidName"
    new_filename = "Validname.jpg"
    new_path = os.path.join(tmpdir, new_filename)

    with patch('src.main.sanitise_filename', return_value="Validname"), \
            patch('os.rename', side_effect=OSError("Permission denied")), \
            patch('builtins.print') as mock_print:
        update_alt_text(image_path, alt_text)

        mock_print.assert_any_call(f"❌ Failed to rename file: Permission denied")
        mock_print.assert_any_call("-" * 100)


@patch('builtins.input', return_value="")
def test_main_no_images(mock_input, mock_find_images, mock_load_blip2_model, mock_generate_caption,
                        mock_update_alt_text):
    mock_find_images.return_value = []
    with patch('builtins.print') as mock_print:
        main()
        mock_print.assert_called_with("⚠️ No images found without alt text in that folder.")


@patch('os.path.exists', return_value=True)  # Mock os.path.exists to always return True
@patch('src.main.find_images_without_alt_text')
@patch('src.main.load_blip2_model')
@patch('src.main.generate_caption')
@patch('src.main.update_alt_text')
@patch('builtins.input', return_value="/path/to/images")
def test_main_with_images(mock_input, mock_update_alt_text, mock_generate_caption,
                          mock_load_blip2_model, mock_find_images, mock_path_exists):
    # Mock find_images_without_alt_text
    mock_find_images.return_value = ["/path/to/images/image1.jpg", "/path/to/images/image2.png"]

    # Mock load_blip2_model
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_device = torch.device("cpu")
    mock_load_blip2_model.return_value = (mock_processor, mock_model, mock_device)

    # Mock generate_caption
    mock_generate_caption.side_effect = ["Caption 1", "Caption 2"]

    # Run the main function
    main()

    # Validate that find_images_without_alt_text was called correctly
    mock_find_images.assert_called_once_with("/path/to/images")

    # Validate that update_alt_text was called with correct arguments
    mock_update_alt_text.assert_any_call("/path/to/images/image1.jpg", "Caption 1")
    mock_update_alt_text.assert_any_call("/path/to/images/image2.png", "Caption 2")
    # Ensure print statements are called appropriately