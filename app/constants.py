from vertexai.generative_models import SafetySetting

# Field definitions for cheques
# Consider defining a Pydantic model for FieldDefinition if more complex attributes are needed
FIELDS = [
    {"id": 1, "name": "bank_name"},
    {"id": 2, "name": "bank_branch"},
    {"id": 3, "name": "account_number"},
    {"id": 4, "name": "date"}, # Expected format YYYY-MM-DD
    {"id": 5, "name": "payee_name"},
    {"id": 6, "name": "amount_words"},
    {"id": 7, "name": "amount_numeric"}, # Should be string representation of number
    {"id": 8, "name": "currency"}, # Expected 3-letter ISO 4217 code
    {"id": 9, "name": "issuer_name"},
    {"id": 10, "name": "signature_present"}, # Expected "YES" or "NO"
    {"id": 11, "name": "IFSC"}, # Expected 11 char alphanumeric, 5th is '0'
    {"id": 12, "name": "micr_scan_instrument_number"}, # 6 digits
    {"id": 13, "name": "micr_scan_payee_details"}, # 9 digits
    {"id": 14, "name": "micr_scan_micr_acno"}, # 6 digits or "000000"
    {"id": 15, "name": "micr_scan_instrument_type"} # 2 digits
]

# Mapping string thresholds from config to Vertex AI Enum
# VertexAI HarmBlockThreshold values:
# UNSPECIFIED, OFF, LOW_AND_ABOVE, MEDIUM_AND_ABOVE, HIGH_AND_ABOVE
# In the SDK, OFF is represented by HarmBlockThreshold.BLOCK_NONE or similar.
# Let's use BLOCK_NONE as per the common SDK usage for "OFF".
threshold_map = {
    "BLOCK_NONE": SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    "BLOCK_LOW_AND_ABOVE": SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    "BLOCK_MEDIUM_AND_ABOVE": SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    "BLOCK_ONLY_HIGH": SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH, # maps to BLOCK_HIGH for older SDKs
}

# Safety settings for Vertex AI
# Note: The original code used SafetySetting.HarmBlockThreshold.OFF.
# The current SDK might prefer HarmBlockThreshold.BLOCK_NONE.
# Assuming OFF implies BLOCK_NONE based on typical behavior.
# This is now more dynamically configured via settings, but kept here for clarity on structure
SAFETY_SETTINGS = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=threshold_map.get("BLOCK_NONE") # settings.HARM_CATEGORY_HATE_SPEECH_THRESHOLD
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=threshold_map.get("BLOCK_NONE") # settings.HARM_CATEGORY_DANGEROUS_CONTENT_THRESHOLD
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=threshold_map.get("BLOCK_NONE") # settings.HARM_CATEGORY_SEXUALLY_EXPLICIT_THRESHOLD
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=threshold_map.get("BLOCK_NONE") # settings.HARM_CATEGORY_HARASSMENT_THRESHOLD
    ),
]

SUPPORTED_MIME_TYPES = {
    '.pdf': 'application/pdf',  # Note: PDF processing for images needs pdf2image
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.tiff': 'image/tiff',
    '.tif': 'image/tiff'
}

# If pdf2image is used, these could be relevant
# PDF_POPPLER_PATH = None # os.getenv("POPPLER_PATH")
# PDF_THREAD_COUNT = 4

TEMP_DIR_PREFIX = "cheque_job_"
EXCEL_FILENAME_TEMPLATE = "cheque_extraction_results_{job_id}.xlsx"
EXCEL_SHEET_NAME_MAX_LEN = 31 # Excel sheet name limit

# Default error signature coordinates structure
DEFAULT_ERROR_SIGNATURE_COORDS = {
    "exists": False, "coordinates": None, "description": "Processing error", "confidence": 0.0
}