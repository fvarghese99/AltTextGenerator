# src/helpers/file_utils.py

import re
import uuid

def sanitise_filename(text: str, max_length: int = 100) -> str:
    """
    Sanitise filenames by removing disallowed characters and truncating.
    If the original text is empty or only whitespace, returns an empty string.
    If the result is empty for non-empty input, returns a fallback using a short UUID.
    """
    original = text
    text = text.strip()
    if not text:
        return ""

    text = re.sub(r'[\\\/:*?"<>|]', '', text)
    text = text.lower()

    # Truncate to max_length preserving whole tokens where possible
    tokens = text.split()
    new_text = ""
    for token in tokens:
        if not new_text:
            if len(token) <= max_length:
                new_text = token
            else:
                new_text = token[:max_length]
        else:
            if len(new_text) + 1 + len(token) <= max_length:
                new_text += " " + token
            else:
                break

    # Capitalise the first letter
    if new_text:
        new_text = new_text[0].upper() + new_text[1:]
    else:
        # Input was not empty originally but all characters were removed
        new_text = str(uuid.uuid4())[:8]
    return new_text