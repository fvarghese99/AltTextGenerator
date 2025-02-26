# tests/test_blip2.py

import pytest
from unittest.mock import patch, MagicMock, call
from src.models.blip2 import load_blip2_model, generate_caption, generate_captions, _preprocess_image_batch
import torch
import io
from PIL import Image


@pytest.fixture
def mock_image_open():
    with patch('src.models.blip2.Image.open') as mock_open:
        yield mock_open


def test_load_blip2_model_cpu(monkeypatch):
    # Mock torch.cuda.is_available to return False
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    # Mock torch.backends.mps.is_available to return False
    monkeypatch.setattr(torch.backends, 'mps', MagicMock())
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)

    # Mock gc.collect to avoid errors
    with patch('gc.collect') as mock_gc, \
            patch('src.models.blip2.Blip2Processor.from_pretrained') as mock_processor, \
            patch('src.models.blip2.Blip2ForConditionalGeneration.from_pretrained') as mock_model:
        mock_processor.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.to = MagicMock(return_value=mock_model.return_value)  # Mock 'to' method

        processor, model, device = load_blip2_model()

        mock_processor.assert_called_once_with("Salesforce/blip2-opt-2.7b")
        mock_model.assert_called_once_with("Salesforce/blip2-opt-2.7b")
        assert device.type == "cpu"
        assert model.to.call_args[0][0] == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_blip2_model_cuda(monkeypatch):
    # This test runs only if CUDA is available
    with patch('gc.collect') as mock_gc, \
            patch('torch.cuda.empty_cache') as mock_empty_cache, \
            patch('src.models.blip2.Blip2Processor.from_pretrained') as mock_processor, \
            patch('src.models.blip2.Blip2ForConditionalGeneration.from_pretrained') as mock_model:
        # Create mock property for CUDA device
        mock_props = MagicMock()
        mock_props.name = "Test GPU"
        mock_props.total_memory = 8 * 1024 ** 3  # 8GB

        monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda x: mock_props)

        mock_processor.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.to = MagicMock(return_value=mock_model.return_value)

        processor, model, device = load_blip2_model()

        # Check that we load with half precision for CUDA
        mock_processor.assert_called_once_with("Salesforce/blip2-opt-2.7b")
        mock_model.assert_called_once_with(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16
        )
        assert device.type == "cuda"
        assert model.to.call_args[0][0] == device


def test_load_blip2_model_mps(monkeypatch):
    # Mock torch.cuda.is_available to return False
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    # Mock torch.backends.mps.is_available to return True
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)

    with patch('gc.collect') as mock_gc, \
            patch('src.models.blip2.Blip2Processor.from_pretrained') as mock_processor, \
            patch('src.models.blip2.Blip2ForConditionalGeneration.from_pretrained') as mock_model:
        mock_processor.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.to = MagicMock(return_value=mock_model.return_value)  # Mock 'to' method

        processor, model, device = load_blip2_model()

        mock_processor.assert_called_once_with("Salesforce/blip2-opt-2.7b")
        mock_model.assert_called_once_with("Salesforce/blip2-opt-2.7b")
        assert device.type == "mps"
        assert model.to.call_args[0][0] == device


def test_preprocess_image_batch():
    # Test the _preprocess_image_batch function with regular images
    mock_image = MagicMock(spec=Image.Image)
    mock_image.mode = 'RGB'
    mock_image.resize.return_value = mock_image

    with patch('PIL.Image.open', return_value=mock_image):
        image_paths = ['/path/to/image1.jpg', '/path/to/image2.png']
        result = _preprocess_image_batch(image_paths)

        assert len(result) == 2
        assert result[0][0] == '/path/to/image1.jpg'
        assert result[1][0] == '/path/to/image2.png'
        assert mock_image.resize.call_count == 2
        mock_image.resize.assert_has_calls([call((512, 512)), call((512, 512))])


def test_preprocess_image_batch_with_svg():
    # Test handling SVG images
    mock_image = MagicMock(spec=Image.Image)
    mock_image.mode = 'RGB'
    mock_image.resize.return_value = mock_image

    with patch('cairosvg.svg2png', return_value=b'fake_png_data'), \
            patch('PIL.Image.open', return_value=mock_image):
        image_paths = ['/path/to/image.svg']
        result = _preprocess_image_batch(image_paths)

        assert len(result) == 1
        assert result[0][0] == '/path/to/image.svg'
        mock_image.resize.assert_called_once_with((512, 512))


def test_preprocess_image_batch_error_handling():
    # Test handling invalid images gracefully
    with patch('PIL.Image.open', side_effect=Exception("Cannot open image")):
        image_paths = ['/path/to/bad_image.jpg']
        result = _preprocess_image_batch(image_paths)

        # Should return empty list as the image couldn't be processed
        assert len(result) == 0


def test_generate_caption():
    # Test the generate_caption function which now uses generate_captions
    processor = MagicMock()
    model = MagicMock()
    device = torch.device("cpu")

    with patch('src.models.blip2.generate_captions') as mock_generate_captions:
        mock_generate_captions.return_value = {'/path/to/image.jpg': 'A test caption'}

        result = generate_caption(processor, model, device, '/path/to/image.jpg')

        mock_generate_captions.assert_called_once_with(
            processor, model, device, ['/path/to/image.jpg'], batch_size=1
        )
        assert result == 'A test caption'


def test_generate_captions_basic():
    # Test the basic functionality of generate_captions
    processor = MagicMock()
    model = MagicMock()
    device = torch.device("cpu")

    mock_image = MagicMock(spec=Image.Image)
    mock_image.mode = 'RGB'
    mock_image.resize.return_value = mock_image

    processor.return_value = {'input_ids': torch.tensor([1, 2, 3])}
    processor.batch_decode.return_value = ['Caption 1', 'Caption 2']
    model.generate.return_value = torch.tensor([4, 5, 6])

    with patch('src.models.blip2._preprocess_image_batch') as mock_preprocess:
        mock_preprocess.return_value = [
            ('/path/to/image1.jpg', mock_image),
            ('/path/to/image2.jpg', mock_image)
        ]

        result = generate_captions(processor, model, device,
                                   ['/path/to/image1.jpg', '/path/to/image2.jpg'],
                                   batch_size=2)

        assert len(result) == 2
        assert '/path/to/image1.jpg' in result
        assert '/path/to/image2.jpg' in result
        assert result['/path/to/image1.jpg'] == 'Caption 1'
        assert result['/path/to/image2.jpg'] == 'Caption 2'


def test_generate_captions_empty():
    # Test with empty image list
    processor = MagicMock()
    model = MagicMock()
    device = torch.device("cpu")

    result = generate_captions(processor, model, device, [])

    assert result == {}


def test_generate_captions_batch_error_fallback():
    # Test the fallback to individual processing when batch processing fails
    processor = MagicMock()
    model = MagicMock()
    device = torch.device("cpu")

    mock_image = MagicMock(spec=Image.Image)
    mock_image.mode = 'RGB'
    mock_image.resize.return_value = mock_image

    # First call raises exception, then individual calls work
    processor.side_effect = [Exception("Batch error"), {'input_ids': torch.tensor([1, 2, 3])}]
    processor.batch_decode.return_value = ['Caption 1']
    model.generate.return_value = torch.tensor([4, 5, 6])

    with patch('src.models.blip2._preprocess_image_batch') as mock_preprocess, \
            patch('builtins.print') as mock_print:
        mock_preprocess.return_value = [('/path/to/image1.jpg', mock_image)]

        # First batch fails, individual processing succeeds
        with patch.object(processor, '__call__',
                          side_effect=[Exception("Batch error"), {'input_ids': torch.tensor([1, 2, 3])}]):
            with patch.object(processor, 'batch_decode', return_value=['Individual caption']):
                result = generate_captions(processor, model, device, ['/path/to/image1.jpg'], batch_size=1)

                # Should report the error but still return a result via individual processing
                mock_print.assert_any_call("Falling back to individual processing...")
                assert len(result) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_generate_captions_cuda_memory_management():
    # This test runs only if CUDA is available
    processor = MagicMock()
    model = MagicMock()
    device = torch.device("cuda")

    mock_image = MagicMock(spec=Image.Image)
    mock_image.mode = 'RGB'
    mock_image.resize.return_value = mock_image

    processor.return_value = {'input_ids': torch.tensor([1, 2, 3])}
    processor.batch_decode.return_value = ['Caption 1']
    model.generate.return_value = torch.tensor([4, 5, 6])

    with patch('src.models.blip2._preprocess_image_batch') as mock_preprocess, \
            patch('torch.cuda.empty_cache') as mock_empty_cache, \
            patch('gc.collect') as mock_gc:
        mock_preprocess.return_value = [('/path/to/image1.jpg', mock_image)]

        result = generate_captions(processor, model, device, ['/path/to/image1.jpg'], batch_size=1)

        # Memory management should be called for CUDA devices
        assert mock_empty_cache.call_count >= 2  # At least at the beginning and end
        assert mock_gc.call_count >= 2
        assert len(result) == 1