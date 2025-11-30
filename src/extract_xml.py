"""Simple XML tag extractor utility."""

import re
from typing import Optional


def extract_xml_tag(text: str, tag_name: str) -> Optional[str]:
    """Extract content from an XML-style tag.
    
    Args:
        text: The text to search in
        tag_name: The tag name (without angle brackets)
    
    Returns:
        The content inside the tag, or None if not found
    """
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None
