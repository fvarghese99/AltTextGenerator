# src/helpers/file_utils.py

import re

def sanitise_filename(text: str, max_length: int = 100) -> str:
    """
    Sanitise filenames by removing disallowed characters and truncating.
    """
    # Remove leading/trailing whitespace and disallowed characters
    text = text.strip()

    # Remove disallowed characters
    text = re.sub(r'[\\\/:*?"<>|]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Truncate to max_length
    tokens = text.split()
    new_text = ""
    for token in tokens:
        if not new_text:
            if len(token) <= max_length:
                new_text = token
            else:
                new_text = token[:max_length]  # Truncate the first token if too long
        else:
            if len(new_text) + 1 + len(token) <= max_length:
                new_text += " " + token
            else:
                break

    # Capitalise the first letter
    if new_text:
        new_text = new_text[0].upper() + new_text[1:]
    return new_text