import time
import json
import random # Added for jitter
import logging
import traceback
from typing import List, Any, Optional, Dict

import vertexai
from vertexai.generative_models import GenerativeModel, Part, ResponseBlockedError
from google.api_core import exceptions as google_exceptions
from pydantic import ValidationError


from app.core.config import settings
from app.constants import SAFETY_SETTINGS
from app.utils.json_utils import extract_json_from_text
from app.api.v1.schemas import SignatureExtractionResult, LLMExtractionOutput, ExtractedFieldData, Coordinates

logger = logging.getLogger(__name__)

class VertexAIService:
    def __init__(self):
        try:
            vertexai.init(project=settings.PROJECT_ID, location=settings.LOCATION, api_endpoint=settings.API_ENDPOINT)
            logger.info(f"Vertex AI initialized for project {settings.PROJECT_ID} in {settings.LOCATION}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}", exc_info=True)
            raise
        # Model can be initialized per call or once if thread-safe and same model used.
        # For now, initialize per call to ensure safety settings are applied correctly.

    def _get_model(self, model_name: str = "gemini-1.5-pro") -> GenerativeModel:
        # Ensure SAFETY_SETTINGS uses the correct enum if configured via strings
        # This is now handled in constants.py with a map
        return GenerativeModel(model_name, safety_settings=SAFETY_SETTINGS)

    def _call_vertex_ai_with_retry(
        self,
        model_instance: GenerativeModel,
        prompt_parts: List[Any],
        max_retries: int = 5,
        initial_delay: float = 1.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ) -> Any: # Returns the model's response object
        num_retries = 0
        delay = initial_delay
        retryable_errors = (
            google_exceptions.ResourceExhausted,    # 429
            google_exceptions.TooManyRequests,      # 429
            google_exceptions.ServiceUnavailable,   # 503
            google_exceptions.DeadlineExceeded,     # 504
            google_exceptions.InternalServerError,  # 500 (sometimes retryable)
            google_exceptions.Aborted               # Can be due to transient issues
        )

        while True:
            try:
                logger.debug(f"Attempting Vertex AI API call (Attempt {num_retries + 1}/{max_retries + 1})")
                response = model_instance.generate_content(prompt_parts, request_options={"timeout": 300}) # Added timeout
                logger.debug(f"Vertex AI API call successful (Attempt {num_retries + 1}/{max_retries + 1})")
                return response
            except ResponseBlockedError as rbe:
                logger.error(f"Vertex AI content generation blocked: {rbe}. Finish reason: {rbe.response.prompt_feedback.block_reason if rbe.response else 'N/A'}")
                # You might want to return a specific structure or raise a custom exception here
                # For now, re-raising to be caught by the caller's generic exception handling
                raise
            except retryable_errors as e:
                num_retries += 1
                if num_retries > max_retries:
                    logger.error(
                        f"Max retries ({max_retries}) exceeded for Vertex AI API call. "
                        f"Last error: {type(e).__name__} - {e}"
                    )
                    raise 
                actual_delay = delay
                if jitter:
                    actual_delay += random.uniform(0, delay * 0.25)
                logger.warning(
                    f"Vertex AI API call failed with {type(e).__name__} (Attempt {num_retries}/{max_retries}). "
                    f"Retrying in {actual_delay:.2f} seconds..."
                )
                time.sleep(actual_delay)
                delay *= exponential_base
            except Exception as e:
                logger.error(f"Non-retryable error during Vertex AI API call: {type(e).__name__} - {e}")
                logger.error(traceback.format_exc())
                raise

    def extract_data_from_document(
        self,
        file_data: bytes,
        file_mime_type: str,
        prompt: str,
        file_path_for_logging: str = "unknown_file"
    ) -> Optional[LLMExtractionOutput]:
        model = self._get_model() # Default "gemini-1.5-pro"
        file_part = Part.from_data(data=file_data, mime_type=file_mime_type)
        
        try:
            response = self._call_vertex_ai_with_retry(model, [prompt, file_part])
            if not response or not response.text:
                logger.error(f"Empty response from Vertex AI for {file_path_for_logging}")
                return None

            json_str = extract_json_from_text(response.text.strip())
            if not json_str:
                logger.error(f"Failed to extract JSON from LLM response for {file_path_for_logging}. Response text: {response.text[:500]}...")
                return LLMExtractionOutput(full_text=response.text, extracted_fields=[]) # Return raw text if JSON fails

            try:
                # Validate and parse the extracted JSON into Pydantic models
                parsed_data = json.loads(json_str)
                # Before creating LLMExtractionOutput, validate individual fields if necessary
                # The current LLMExtractionOutput schema will validate its structure.
                # Individual ExtractedFieldData items are validated by their own schema.
                
                # Manually reconstruct ExtractedFieldData to trigger Pydantic validation
                validated_fields = []
                if 'extracted_fields' in parsed_data and isinstance(parsed_data['extracted_fields'], list):
                    for field_dict in parsed_data['extracted_fields']:
                        try:
                            validated_fields.append(ExtractedFieldData(**field_dict))
                        except ValidationError as ve:
                            logger.warning(f"Validation failed for field '{field_dict.get('field_name', 'Unknown')}' in {file_path_for_logging}: {ve}. Raw value: '{field_dict.get('value')}'")
                            # Keep the field but with potentially null value and reason
                            validated_fields.append(ExtractedFieldData(
                                field_name=field_dict.get('field_name', 'unknown_field_name'),
                                value=field_dict.get('value'), # Keep original value if validation fails, schema might coerce/error
                                confidence=field_dict.get('confidence', 0.1), # Lower confidence
                                text_segment=field_dict.get('text_segment'),
                                reason=f"Pydantic Validation Error: {str(ve)[:100]}",
                                language=field_dict.get('language')
                            ))
                
                return LLMExtractionOutput(
                    full_text=parsed_data.get("full_text", response.text if not parsed_data.get("full_text") else None),
                    extracted_fields=validated_fields
                )

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error for {file_path_for_logging}: {e}. JSON string: {json_str[:500]}...")
                return LLMExtractionOutput(full_text=response.text, extracted_fields=[])
            except ValidationError as ve:
                logger.error(f"Pydantic validation error for LLMExtractionOutput for {file_path_for_logging}: {ve}. JSON string: {json_str[:500]}...")
                # Attempt to return partial data if possible, or a structured error
                return LLMExtractionOutput(
                    full_text=json.loads(json_str).get("full_text") if json_str else response.text,
                    extracted_fields=[], # Or try to parse individual fields with try-except
                    # error_message=f"Pydantic validation error: {str(ve)}" # If schema supported error message
                )

        except ResponseBlockedError: # Specific handling if needed, already logged in retry_call
             logger.error(f"Content generation blocked by Vertex AI for {file_path_for_logging}.")
             return LLMExtractionOutput(full_text="Response blocked by API.", extracted_fields=[])
        except Exception as e:
            logger.error(f"Failed to process document with Vertex AI for {file_path_for_logging}: {e}", exc_info=True)
            return None


    def extract_signature_coordinates(
        self,
        file_data: bytes,
        file_mime_type: str,
        prompt: str,
        file_path_for_logging: str = "unknown_file"
    ) -> SignatureExtractionResult:
        model = self._get_model()
        file_part = Part.from_data(data=file_data, mime_type=file_mime_type)
        default_error_response = SignatureExtractionResult(
            exists=False, coordinates=None, description="Failed due to API or parsing error", confidence=0.0
        )

        try:
            response = self._call_vertex_ai_with_retry(model, [prompt, file_part])
            if not response or not response.text:
                logger.error(f"Empty signature response from Vertex AI for {file_path_for_logging}")
                return default_error_response

            json_str = extract_json_from_text(response.text.strip())
            if not json_str:
                logger.error(f"Failed to extract JSON from signature response for {file_path_for_logging}. Text: {response.text[:200]}")
                return default_error_response
            
            try:
                # Pydantic will validate the structure here
                sig_data = json.loads(json_str)
                if sig_data.get("coordinates") and isinstance(sig_data["coordinates"], dict):
                     # Ensure coordinates are valid before creating the model
                    coords = Coordinates(**sig_data["coordinates"])
                    sig_data["coordinates"] = coords
                
                return SignatureExtractionResult(**sig_data)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"JSON parsing or Pydantic validation error for signature for {file_path_for_logging}: {e}. JSON string: {json_str[:200]}...")
                return SignatureExtractionResult(
                    exists=False, coordinates=None, description=f"JSON/Validation Error: {str(e)[:100]}", confidence=0.0
                )

        except ResponseBlockedError:
             logger.error(f"Signature Content generation blocked by Vertex AI for {file_path_for_logging}.")
             return SignatureExtractionResult(exists=False, confidence=0.0, description="Response blocked by API")
        except Exception as e:
            logger.error(f"Failed to extract signature coordinates for {file_path_for_logging}: {e}", exc_info=True)
            return default_error_response