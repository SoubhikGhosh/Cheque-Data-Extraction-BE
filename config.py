# /cheque_extraction_api/config.py

from vertexai.generative_models import SafetySetting

# --- Vertex AI Configuration ---
GCP_PROJECT_ID = "hbl-uat-ocr-fw-app-prj-spk-4d"
GCP_LOCATION = "asia-south1"
API_ENDPOINT = "asia-south1-aiplatform.googleapis.com"
MODEL_NAME = "gemini-2.5-flash"

# --- Performance Configuration ---
# Maximum number of concurrent threads to process images.
MAX_WORKERS = 120
# Number of images to process in a single batch.
BATCH_SIZE = 60

# --- Vertex AI Generation Configuration ---
# This configuration ensures the model provides a more deterministic
# and structured JSON output.
# - temperature=0.0 makes the output less random.
# - response_mime_type="application/json" forces the model to output a JSON object.
GENERATION_CONFIG = {
    "temperature": 0.0,
    "response_mime_type": "application/json",
}

# --- Application Configuration ---
# Defines the fields to be extracted from the cheques.

OUTPUT_DIR = "job_outputs" 

FIELDS = [
    {"id": 1, "name": "bank_name"},
    {"id": 2, "name": "bank_branch"},
    {"id": 3, "name": "account_number"},
    {"id": 4, "name": "date"},
    {"id": 5, "name": "payee_name"},
    {"id": 6, "name": "amount_words"},
    {"id": 7, "name": "amount_numeric"},
    {"id": 8, "name": "currency"},
    {"id": 9, "name": "issuer_name"},
    {"id": 10, "name": "IFSC"},
    {"id": 11, "name": "micr_scan_instrument_number"},
    {"id": 12, "name": "micr_scan_payee_details"},
    {"id": 13, "name": "micr_scan_micr_acno"},
    {"id": 14, "name": "micr_scan_instrument_type"}
]

# --- Vertex AI Safety Settings ---
# Configuration to disable content safety filters for this specific use case.
SAFETY_SETTINGS = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]