# tests/conftest.py

import pytest
from unittest.mock import patch

@pytest.fixture
def mock_image_open():
    with patch('src.models.blip2.Image.open') as mock_open:
        yield mock_open