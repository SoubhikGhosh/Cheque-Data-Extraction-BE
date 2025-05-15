import re
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator, field_validator
from datetime import date as DateType # Alias to avoid confusion with field name 'date'

# Validators
def validate_date_format(value: Optional[str]) -> Optional[str]:
    if value is None or value == "" or value.lower() == "not found":
        return value
    try:
        DateType.fromisoformat(value) # Expects YYYY-MM-DD
        return value
    except (ValueError, TypeError):
        raise ValueError("Date must be in YYYY-MM-DD format")

def validate_ifsc_code(value: Optional[str]) -> Optional[str]:
    if value is None or value == "" or value.lower() == "not found" or value.lower() == "error":
        return value
    if not isinstance(value, str) or not re.fullmatch(r"^[A-Z]{4}0[A-Z0-9]{6}$", value.upper()):
        raise ValueError("IFSC code must be 11 characters, first 4 alphabetic, 5th is zero, last 6 alphanumeric")
    return value.upper()

def validate_micr_digits(length: int):
    def validator_func(value: Optional[str]) -> Optional[str]:
        if value is None or value == "" or value.lower() == "not found":
            return value
        # Special case for micr_scan_micr_acno which can be "000000"
        if length == 6 and value == "000000" and "micr_acno" in validator_func.__qualname__: # Heuristic to check field
             return value
        if not isinstance(value, str) or not re.fullmatch(rf"^\d{{{length}}}$", value):
            raise ValueError(f"MICR field must be exactly {length} digits")
        return value
    return validator_func

def validate_currency_code(value: Optional[str]) -> Optional[str]:
    if value is None or value == "" or value.lower() == "not found":
        return value
    if not isinstance(value, str) or not re.fullmatch(r"^[A-Z]{3}$", value.upper()):
        raise ValueError("Currency code must be 3 uppercase alphabetic characters (ISO 4217)")
    return value.upper()

def validate_signature_present(value: Optional[str]) -> Optional[str]:
    if value is None or value == "" or value.lower() == "not found":
        return value
    if value.upper() not in ["YES", "NO"]:
        raise ValueError("signature_present must be 'YES' or 'NO'")
    return value.upper()

def validate_amount_numeric(value: Optional[str]) -> Optional[str]:
    if value is None or value == "" or value.lower() == "not found":
        return value
    # Allows digits, optional decimal point, and up to two decimal places.
    # Allows for values like "1500", "1500.00", "1500.50"
    if not isinstance(value, str) or not re.fullmatch(r"^\d+(\.\d{1,2})?$", value):
        raise ValueError("Amount numeric must be a valid number string, e.g., '1500' or '1500.50'")
    return value


class Coordinates(BaseModel):
    x1: float = Field(..., ge=0.0, le=1.0)
    y1: float = Field(..., ge=0.0, le=1.0)
    x2: float = Field(..., ge=0.0, le=1.0)
    y2: float = Field(..., ge=0.0, le=1.0)

class SignatureExtractionResult(BaseModel):
    exists: bool
    coordinates: Optional[Coordinates] = None
    description: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)

class ExtractedFieldData(BaseModel):
    field_name: str
    value: Optional[Union[str, bool, None]] = None # Allow None for not found/error
    confidence: float = Field(..., ge=0.0, le=1.0)
    text_segment: Optional[str] = None
    reason: Optional[str] = None
    language: Optional[str] = None

    @field_validator('value', mode='before')
    @classmethod
    def validate_field_value(cls, v, info):
        field_name = info.data.get('field_name')
        if field_name == 'date':
            return validate_date_format(v)
        elif field_name == 'IFSC':
            return validate_ifsc_code(v)
        elif field_name == 'micr_scan_instrument_number':
            return validate_micr_digits(6)(v)
        elif field_name == 'micr_scan_payee_details':
            return validate_micr_digits(9)(v)
        elif field_name == 'micr_scan_micr_acno':
            # Custom validator for micr_scan_micr_acno as it can be "000000"
            if v is None or v == "" or v.lower() == "not found": return v
            if v == "000000": return v
            if not isinstance(v, str) or not re.fullmatch(r"^\d{6}$", v):
                raise ValueError("micr_scan_micr_acno must be 6 digits or '000000'")
            return v
        elif field_name == 'micr_scan_instrument_type':
            return validate_micr_digits(2)(v)
        elif field_name == 'currency':
            return validate_currency_code(v)
        elif field_name == 'signature_present':
            return validate_signature_present(str(v) if isinstance(v, bool) else v) # Handle potential boolean from LLM
        elif field_name == 'amount_numeric':
            return validate_amount_numeric(v)
        # Add other field-specific validations if needed
        return v

class ChequePageData(BaseModel):
    page_num: int
    text: Optional[str] = None

class LLMExtractionOutput(BaseModel):
    full_text: Optional[str] = None
    extracted_fields: List[ExtractedFieldData] = []

class MultimodalProcessingResult(BaseModel):
    file_path: Optional[str] = None # Added for tracking in combined results
    text: Optional[str] = None
    pages: List[ChequePageData] = []
    extracted_fields: List[ExtractedFieldData] = []
    signature_coordinates: Optional[SignatureExtractionResult] = None
    error: Optional[str] = None # For capturing processing errors for a file

# API Response Models
class FileUploadResponse(BaseModel):
    status: str
    job_id: str
    message: str
    files: List[str]
    timestamp: int

class JobStatusDetail(BaseModel):
    status: str
    job_id: Optional[str] = None # Make optional as it might not be set on initial error
    total_files: Optional[int] = None
    processed_files: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    progress_percentage: Optional[float] = None
    estimated_time_remaining: Optional[Union[float, str]] = None # Can be seconds or formatted string
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None # For detailed debugging, perhaps not for client
    output_file_path: Optional[str] = None # For completed jobs
    input_files: Optional[List[str]] = None

# For Excel generation, structure per file
class ExcelRow(BaseModel):
    filepath: str
    # Dynamically add other fields based on FIELDS constant
    # e.g., bank_name: Optional[str], bank_name_conf: Optional[float], ...

    model_config = {
        "extra": "allow"  # Allow dynamic fields
    }