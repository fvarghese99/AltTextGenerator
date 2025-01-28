# tests/test_blip2.py

import pytest
from unittest.mock import patch, MagicMock
from src.models.blip2 import load_blip2_model, generate_caption
import torch


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

    with patch('src.models.blip2.Blip2Processor.from_pretrained') as mock_processor, \
            patch('src.models.blip2.Blip2ForConditionalGeneration.from_pretrained') as mock_model:
        mock_processor.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.to = MagicMock(return_value=mock_model.return_value)  # Mock 'to' method

        processor, model, device = load_blip2_model()

        mock_processor.assert_called_once_with("Salesforce/blip2-opt-2.7b")
        mock_model.assert_called_once_with("Salesforce/blip2-opt-2.7b")
        assert device.type == "cpu"
        assert model.to.call_args[0][0] == device


def test_load_blip2_model_cuda(monkeypatch):
    # Mock torch.cuda.is_available to return True
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)

    with patch('src.models.blip2.Blip2Processor.from_pretrained') as mock_processor, \
            patch('src.models.blip2.Blip2ForConditionalGeneration.from_pretrained') as mock_model:
        mock_processor.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.to = MagicMock(return_value=mock_model.return_value)  # Mock 'to' method

        processor, model, device = load_blip2_model()

        mock_processor.assert_called_once_with("Salesforce/blip2-opt-2.7b")
        mock_model.assert_called_once_with("Salesforce/blip2-opt-2.7b")
        assert device.type == "cuda"
        assert model.to.call_args[0][0] == device


def test_load_blip2_model_mps(monkeypatch):
    # Mock torch.cuda.is_available to return False
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    # Mock torch.backends.mps.is_available to return True
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)

    with patch('src.models.blip2.Blip2Processor.from_pretrained') as mock_processor, \
            patch('src.models.blip2.Blip2ForConditionalGeneration.from_pretrained') as mock_model:
        mock_processor.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.to = MagicMock(return_value=mock_model.return_value)  # Mock 'to' method

        processor, model, device = load_blip2_model()

        mock_processor.assert_called_once_with("Salesforce/blip2-opt-2.7b")
        mock_model.assert_called_once_with("Salesforce/blip2-opt-2.7b")
        assert device.type == "mps"
        assert model.to.call_args[0][0] == device


@patch('src.models.blip2.Image.open')
def test_generate_caption(mock_image_open, monkeypatch):
    # Mock Image.open().convert().resize()
    mock_image = MagicMock()
    mock_image_open.return_value = mock_image
    mock_image.convert.return_value = mock_image
    mock_image.resize.return_value = mock_image

    # Mock processor
    mock_processor = MagicMock()
    mock_processor.batch_decode.return_value = ["A caption"]

    # Mock model.generate
    mock_model = MagicMock()
    mock_model.generate.return_value = ["generated_tokens"]

    # Mock device
    device = torch.device("cpu")

    caption = generate_caption(mock_processor, mock_model, device, "path/to/image.jpg")

    mock_image_open.assert_called_once_with("path/to/image.jpg")
    mock_image.convert.assert_called_once_with("RGB")
    mock_image.resize.assert_called_once_with((512, 512))
    mock_processor.assert_called_once_with(images=mock_image, return_tensors="pt")
    mock_model.generate.assert_called_once()
    mock_processor.batch_decode.assert_called_once_with(["generated_tokens"], skip_special_tokens=True)
    assert caption == "A caption"


def test_generate_caption_no_captions(mock_image_open, monkeypatch):
    # Similar to the above test but with no captions
    with patch('src.models.blip2.Image.open') as mock_image_open, \
            patch('src.models.blip2.Blip2Processor') as mock_processor_class, \
            patch('src.models.blip2.Blip2ForConditionalGeneration') as mock_model_class:
        mock_image = MagicMock()
        mock_image_open.return_value = mock_image
        mock_image.convert.return_value = mock_image
        mock_image.resize.return_value = mock_image

        mock_processor = MagicMock()
        mock_processor.batch_decode.return_value = []

        mock_model = MagicMock()
        mock_model.generate.return_value = []

        caption = generate_caption(mock_processor, mock_model, torch.device("cpu"), "path/to/image.jpg")
        assert caption == ""