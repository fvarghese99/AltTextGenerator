# tests/test_file_utils.py

import pytest
from src.helpers.file_utils import sanitise_filename

@pytest.mark.parametrize(
    "input_text, max_length, expected",
    [
        ("Valid Filename", 100, "Valid filename"),
        ("   Leading and trailing spaces   ", 100, "Leading and trailing spaces"),
        ("Invalid/Characters\\:*?\"<>|", 100, "Invalidcharacters"),
        (
            "ThisIsAReallyLongFilenameThatExceedsTheMaximumLengthAllowedForTestingPurposesExtraChars1234567890ExtraChars1234567890", 100, "Thisisareallylongfilenamethatexceedsthemaximumlengthallowedfortestingpurposesextrachars1234567890ext",
        ),
        ("Mixed CASE Letters", 100, "Mixed case letters"),
        ("", 100, ""),
        ("    ", 100, ""),
        ("Short", 5, "Short"),
        ("ExactLengthFilenameHere", 23, "Exactlengthfilenamehere"),
        ("ExactLengthFilenameHere", 22, "Exactlengthfilenameher"),
    ],
)
def test_sanitise_filename(input_text, max_length, expected):
    assert sanitise_filename(input_text, max_length) == expected