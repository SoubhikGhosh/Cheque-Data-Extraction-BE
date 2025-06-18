# /cheque_extraction_api/utils.py

import logging
import time
import random
import json
import re
import traceback
from typing import List, Any, Optional

from vertexai.generative_models import GenerativeModel
from google.api_core import exceptions as google_exceptions
from dateutil import parser

logger = logging.getLogger(__name__)

# --- NEW: Date Parsing Function ---
def parse_and_format_date(date_str: Optional[str]) -> Optional[str]:
    """
    Parses a date string in almost any format and returns it as YYYY-MM-DD.
    Uses dayfirst=True, assuming DD/MM/YYYY for ambiguous dates like 01/02/2024.

    Args:
        date_str: The raw date string from the LLM.

    Returns:
        A formatted date string "YYYY-MM-DD" or the original string if parsing fails.
    """
    if not date_str or not isinstance(date_str, str):
        return date_str  # Return original if null, not a string, or empty

    try:
        # The dayfirst=True flag is crucial for Indian/European date formats
        # It correctly interprets "04/05/2024" as May 4th, not April 5th.
        parsed_date = parser.parse(date_str, dayfirst=True)
        return parsed_date.strftime('%Y-%m-%d')
    except (parser.ParserError, TypeError):
        # If dateutil can't parse it, it might be a non-date string like "Not Found"
        logger.warning(f"Could not parse date: '{date_str}'. Returning original value.")
        return date_str

# --- NEW: Amount Sanitization Function ---
def sanitize_amount(amount_str: Optional[str]) -> Optional[str]:
    """
    Cleans an amount string to be a valid number.
    - Removes all non-digit and non-period characters (like ₹, ,, -).
    - Handles cases with multiple periods by keeping only the last one as a decimal separator.

    Args:
        amount_str: The raw amount string from the LLM.

    Returns:
        A cleaned numeric string or the original string if it's not processable.
    """
    if not amount_str or not isinstance(amount_str, str):
        return amount_str # Return original if null, not a string, or empty

    # Remove anything that is not a digit or a dot
    cleaned_str = re.sub(r'[^\d.]', '', amount_str)
    
    # If multiple dots are present (e.g., "1.50.000.00"), re-assemble the number correctly.
    # This joins all parts before the last dot and appends the last part.
    if cleaned_str.count('.') > 1:
        parts = cleaned_str.split('.')
        cleaned_str = "".join(parts[:-1]) + "." + parts[-1]
        
    return cleaned_str

# --- Existing functions below (no changes needed) ---

def configure_logging():
    """Configures enhanced application-wide logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('google.auth').setLevel(logging.WARNING)
    logger.info("Logging configured.")


def call_vertex_ai_with_retry(
    model_instance: GenerativeModel,
    prompt_parts: List[Any],
    max_retries: int = 5,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0
) -> Any:
    """
    Calls the Vertex AI model with an exponential backoff retry mechanism.
    
    Args:
        model_instance: The initialized GenerativeModel instance.
        prompt_parts: List of parts to send to the model.
        max_retries: Maximum number of retries.
        initial_delay: Initial delay in seconds.
        exponential_base: Multiplier for the delay.
        
    Returns:
        The response from model.generate_content().
        
    Raises:
        google_exceptions.GoogleAPICallError: If retries fail.
    """
    retryable_errors = (
        google_exceptions.ResourceExhausted,
        google_exceptions.ServiceUnavailable,
        google_exceptions.DeadlineExceeded,
    )
    delay = initial_delay
    for i in range(max_retries):
        try:
            return model_instance.generate_content(prompt_parts)
        except retryable_errors as e:
            logger.warning(f"Vertex AI API call failed (Attempt {i + 1}/{max_retries}) with {type(e).__name__}. Retrying in {delay:.2f}s...")
            time.sleep(delay + random.uniform(0, 0.25)) # Add jitter
            delay *= exponential_base
        except Exception as e:
            logger.error(f"Non-retryable error during Vertex AI call: {e}")
            logger.error(traceback.format_exc())
            raise
    
    raise google_exceptions.RetryError(f"Max retries ({max_retries}) exceeded for Vertex AI API call.", None)


def extract_json_from_text(text: str) -> str:
    """
    Extracts a JSON object from a string, even if it's embedded in markdown.
    """
    # Find JSON within markdown code blocks ```json ... ```
    match = re.search(r'```json\s*(\{[\s\S]*\})\s*```', text)
    if match:
        return match.group(1).strip()
    
    # If no markdown, find the first '{' and the last '}'
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1].strip()
        
    # As a last resort, return the text, hoping it's valid JSON
    logger.warning("Could not find clear JSON markers (markdown or brackets), returning raw text.")
    return text