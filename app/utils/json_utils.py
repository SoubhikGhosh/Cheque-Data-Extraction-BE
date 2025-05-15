import re
import json
import logging

logger = logging.getLogger(__name__)

def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract valid JSON from potentially messy text that might contain
    markdown code blocks, explanations, etc.
    Returns the JSON string or None if not found.
    """
    if not text or not isinstance(text, str):
        return None

    # Attempt to find JSON within markdown code blocks first
    json_markdown_pattern = r'```json\s*(\{[\s\S]*?\})\s*```'
    match = re.search(json_markdown_pattern, text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            json.loads(json_str) # Validate if it's actual JSON
            return json_str
        except json.JSONDecodeError:
            logger.warning(f"Found markdown JSON block, but it's invalid: {json_str[:200]}")
            # Fall through to other methods if markdown block is invalid

    # If no valid markdown, try to find the first '{' and last '}' pair that forms valid JSON
    # This is a bit more robust than simple find, tries to parse substring
    try:
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            potential_json = text[first_brace : last_brace + 1]
            json.loads(potential_json) # Validate
            return potential_json.strip()
    except json.JSONDecodeError:
        logger.debug(f"Greedy brace matching failed for JSON parsing from text: {text[:200]}")

    # Final attempt: More complex regex to find a valid JSON object structure
    # This regex tries to match balanced braces.
    # It's not foolproof for all edge cases of malformed text around JSON, but often works.
    json_object_pattern = r'(\{[\s\S]*?\})(?=\s*(?:```|\Z)|[^`])' # Looks for {..} not followed by json```
                                                               # or {..} at the end of string.
                                                               # or {..} followed by non-backtick
    
    # Prioritize matches that are likely intended JSON blocks (e.g., not deeply nested in text)
    # This is heuristic; perfect extraction from arbitrary text is very hard.
    for m in re.finditer(json_object_pattern, text, re.DOTALL):
        potential_json_str = m.group(1).strip()
        try:
            json.loads(potential_json_str)
            logger.debug(f"Found JSON using complex regex pattern: {potential_json_str[:200]}")
            return potential_json_str
        except json.JSONDecodeError:
            continue # Try next match

    logger.warning(f"Could not extract valid JSON from text: {text[:200]}...")
    return None