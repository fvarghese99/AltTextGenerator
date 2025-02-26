# tests/test_main.py

import pytest
import re
from unittest.mock import patch, MagicMock, call
import torch
import os
import time
from src.main import update_alt_text, batch_update_alt_text, main, _get_terminal_width


@pytest.fixture
def mock_find_images():
    with patch('src.main.find_images_without_alt_text') as mock_func:
        yield mock_func


@pytest.fixture
def mock_load_blip2_model():
    with patch('src.main.load_blip2_model') as mock_func:
        yield mock_func


@pytest.fixture
def mock_generate_captions():
    with patch('src.main.generate_captions') as mock_func:
        yield mock_func


@pytest.fixture
def mock_update_alt_text():
    with patch('src.main.update_alt_text') as mock_func:
        yield mock_func


def test_get_terminal_width():
    # Test with shutil available
    with patch('shutil.get_terminal_size', return_value=os.terminal_size((120, 30))):
        assert _get_terminal_width() == 100  # Should cap at 100

    with patch('shutil.get_terminal_size', return_value=os.terminal_size((80, 30))):
        assert _get_terminal_width() == 80  # Should return actual width if < 100

    # Test fallback when shutil raises exception
    with patch('shutil.get_terminal_size', side_effect=OSError("No terminal")):
        assert _get_terminal_width() == 100  # Should fallback to 100


@patch('shutil.get_terminal_size', return_value=os.terminal_size((100, 24)))
def test_update_alt_text_success(mock_terminal_size, tmpdir):
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


@patch('shutil.get_terminal_size', return_value=os.terminal_size((100, 24)))
def test_update_alt_text_no_alt_text(mock_terminal_size, tmpdir):
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


@patch('shutil.get_terminal_size', return_value=os.terminal_size((100, 24)))
def test_update_alt_text_rename_failure(mock_terminal_size, tmpdir):
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


def test_batch_update_alt_text():
    # Test batch_update_alt_text with multiple images
    captions = {
        "/path/to/image1.jpg": "Caption 1",
        "/path/to/image2.png": "Caption 2",
        "/path/to/image3.gif": "Caption 3"
    }

    with patch('src.main.update_alt_text') as mock_update:
        batch_update_alt_text(captions)

        assert mock_update.call_count == 3
        mock_update.assert_has_calls([
            call("/path/to/image1.jpg", "Caption 1"),
            call("/path/to/image2.png", "Caption 2"),
            call("/path/to/image3.gif", "Caption 3")
        ], any_order=True)


def test_batch_update_alt_text_empty():
    # Test with empty dictionary
    with patch('src.main.update_alt_text') as mock_update:
        batch_update_alt_text({})
        mock_update.assert_not_called()


@patch('builtins.input', return_value="")
@patch('os.path.exists', return_value=True)
def test_main_no_images(mock_exists, mock_input, mock_find_images, mock_load_blip2_model):
    mock_find_images.return_value = []  # no images
    with patch('builtins.print') as mock_print:
        main()
        mock_print.assert_any_call("⚠️ No images found without alt text in that folder.")


@patch('time.time', side_effect=[100, 110, 120, 130, 140])  # Mock timestamps
@patch('os.path.exists', return_value=True)
@patch('os.walk')
def test_main_processing_workflow(mock_walk, mock_exists, mock_time,
                                  mock_find_images, mock_load_blip2_model,
                                  mock_generate_captions):
    # Set up mocks for a complete workflow test
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_device = torch.device("cpu")
    mock_load_blip2_model.return_value = (mock_processor, mock_model, mock_device)

    mock_find_images.return_value = [
        "/path/to/image1.jpg",
        "/path/to/image2.png"
    ]

    # No more images to verify
    mock_walk.return_value = []

    # Mock generate_captions to return a dictionary of results
    mock_generate_captions.return_value = {
        "/path/to/image1.jpg": "Caption 1",
        "/path/to/image2.png": "Caption 2"
    }

    with patch('builtins.input', return_value="/path/to/images"), \
            patch('src.main.batch_update_alt_text') as mock_batch_update, \
            patch('builtins.print') as mock_print:

        main()

        # Verify the basic workflow
        mock_find_images.assert_called_once_with("/path/to/images")
        mock_load_blip2_model.assert_called_once()
        mock_generate_captions.assert_called_once_with(mock_processor, mock_model, mock_device,
                                                       ["/path/to/image1.jpg", "/path/to/image2.png"])

        # Verify batch update was called
        mock_batch_update.assert_called_once_with({
            "/path/to/image1.jpg": "Caption 1",
            "/path/to/image2.png": "Caption 2"
        })

        # Verify final stats were printed - using regex to match the pattern without exact formatting
        found_match = False
        pattern = re.compile(r"\n🎉 All done! Processed 2 images in \d+\.\d+s")
        for call_args in mock_print.call_args_list:
            args, _ = call_args
            if args and isinstance(args[0], str) and pattern.match(args[0]):
                found_match = True
                break

        assert found_match, "Could not find message matching the pattern for 'All done' message"


@patch('time.time', side_effect=[100, 110, 120, 130, 140, 150])  # Mock timestamps
@patch('os.path.exists', return_value=True)
def test_main_with_verification(mock_exists, mock_time,
                                mock_find_images, mock_load_blip2_model,
                                mock_generate_captions):
    # Set up mocks for testing the verification workflow
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_device = torch.device("cpu")
    mock_load_blip2_model.return_value = (mock_processor, mock_model, mock_device)

    # Initial images without alt text
    mock_find_images.return_value = ["/path/to/image1.jpg"]

    # Initial captions
    initial_captions = {"/path/to/image1.jpg": "Caption 1"}

    # Verification captions
    verification_captions = {"/path/to/image3.jpg": "Caption 3"}

    # Set up mock to return different results for each call
    mock_generate_captions.side_effect = [initial_captions, verification_captions]

    with patch('builtins.input', return_value="/path/to/images"), \
            patch('src.main.batch_update_alt_text') as mock_batch_update, \
            patch('os.walk') as mock_walk, \
            patch('builtins.print') as mock_print:

        # Set up mock for os.walk to find additional images
        mock_walk.return_value = [
            ("/path/to", [], ["image3.jpg"])  # Additional image found
        ]

        main()

        # Verify both initial and verification processing
        assert mock_generate_captions.call_count == 2
        assert mock_batch_update.call_count == 2

        # First batch update (initial processing)
        mock_batch_update.assert_any_call(initial_captions)

        # Second batch update (verification)
        mock_batch_update.assert_any_call(verification_captions)

        # Use regex pattern to check for the final stats
        found_match = False
        pattern = re.compile(r"\n🎉 All done! Processed 2 images in \d+\.\d+s")
        for call_args in mock_print.call_args_list:
            args, _ = call_args
            if args and isinstance(args[0], str) and pattern.match(args[0]):
                found_match = True
                break

        assert found_match, "Could not find message matching the pattern for 'All done' message"


@patch('shutil.get_terminal_size', return_value=os.terminal_size((100, 24)))
def test_update_alt_text_collision(mock_terminal_size, tmpdir):
    image_path = os.path.join(tmpdir, "image.jpg")
    with open(image_path, 'w') as f:
        f.write("fake image content")

    alt_text = "Collision"
    safe_name = "Collision"

    # Create an existing file with the same name "Collision.jpg"
    existing_file = os.path.join(tmpdir, "Collision.jpg")
    with open(existing_file, 'w') as f:
        f.write("already exists")

    # Next rename candidate would be "Collision_1.jpg"
    new_path = os.path.join(tmpdir, "Collision_1.jpg")

    with patch('src.main.sanitise_filename', return_value=safe_name), \
            patch('builtins.print') as mock_print:
        update_alt_text(image_path, alt_text)

        # The code should skip "Collision.jpg" (since it already exists)
        # and rename to "Collision_1.jpg"
        assert os.path.exists(new_path)
        mock_print.assert_any_call(f"✅ Renamed image to: {new_path}")


@patch('time.time', side_effect=[100, 110, 120, 130, 140])  # Mock timestamps
@patch('os.path.exists', return_value=True)
def test_main_cuda_tips(mock_exists, mock_time,
                        mock_find_images, mock_load_blip2_model,
                        mock_generate_captions):
    # Test that CUDA tips are printed when using CUDA
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_device = torch.device("cuda")  # Using CUDA
    mock_load_blip2_model.return_value = (mock_processor, mock_model, mock_device)

    mock_find_images.return_value = ["/path/to/image1.jpg"]
    mock_generate_captions.return_value = {"/path/to/image1.jpg": "Caption 1"}

    with patch('builtins.input', return_value="/path/to/images"), \
            patch('src.main.batch_update_alt_text'), \
            patch('os.walk', return_value=[]), \
            patch('builtins.print') as mock_print:
        main()

        # Verify CUDA tips were printed
        mock_print.assert_any_call("\n💡 CUDA Performance Note: If you experienced slow processing, try these tips:")
        mock_print.assert_any_call("  - Close other CUDA applications")
        mock_print.assert_any_call("  - Update your NVIDIA drivers")