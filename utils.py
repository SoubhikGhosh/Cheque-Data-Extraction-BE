# /cheque_extraction_api/utils.py

import logging
import time
import random
import json
import re
import traceback
from typing import List, Any

from vertexai.generative_models import GenerativeModel
from google.api_core import exceptions as google_exceptions

logger = logging.getLogger(__name__)

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