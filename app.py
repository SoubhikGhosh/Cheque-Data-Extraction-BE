from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part
import os
import zipfile
import io
import logging
import tempfile
import time
from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd
import uuid
import shutil
import json
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import re
import uvicorn
from collections import Counter
import concurrent.futures
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Adjust logging configuration to ensure detailed logging
def configure_enhanced_logging():
    """Configure logging to capture more detailed information."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('google.auth').setLevel(logging.WARNING)

# Call this at the start of your application
configure_enhanced_logging()

# Initialize FastAPI app
app = FastAPI(
    title="Cheque Data Extraction API",
    description="API for processing zip files containing cheque images using Vertex AI",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vertex AI Configuration
project = "hbl-uat-ocr-fw-app-prj-spk-4d"
vertexai.init(project=project, location="asia-south1", api_endpoint='asia-south1-aiplatform.googleapis.com')

# Safety settings
safety_settings = [
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

# Field definitions for cheques
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
    {"id": 10, "name": "micr_code"},
    {"id": 11, "name": "signature_present"},
    {"id": 12, "name": "IFSC"}
]

# ============ NEW PERFORMANCE OPTIMIZATION CONSTANTS ============
MAX_WORKERS =80
BATCH_SIZE = 40

# Create a thread pool executor at the module level with increased workers
executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Keep a dictionary of futures for tracking
active_tasks = {}
processed_jobs = {}

class ChequeProcessor:
    """Helper class for cheque processing operations using Vertex AI's multimodal capabilities"""
                  
    @staticmethod
    def extract_signature_coordinates(file_data: bytes, file_type: str) -> Dict[str, Any]:
        """
        Extract signature coordinates using Vertex AI's multimodal capabilities
        """
        try:
            # Initialize Vertex AI model
            model = GenerativeModel("gemini-1.5-pro", safety_settings=safety_settings)
            
            # Create a Vertex AI Part from the file data
            file_part = Part.from_data(data=file_data, mime_type=file_type)
            
            # Signature extraction prompt
            signature_prompt = """
            You are a forensic document expert specializing in signature detection.
            
            CRITICAL TASK: Locate the EXACT position of the signature on this Indian bank cheque with maximum precision.
            
            # BANK IDENTIFICATION GUIDE:
            
            1. HDFC BANK:
            - Logo appears in top-left corner
            - Look for "HDFC BANK" text and IFSC code
            - Signature is in bottom-right corner with "Authorised Signatories" text
            - May include "For [COMPANY NAME]" above signature
            - Often has wavy pattern on left side of cheque
            
            2. ICICI BANK:
            - Logo appears in top-left with "ICICI Bank" text
            - May have "Privilege" or "Imperia" premium banking logo
            - Signature appears bottom-right with "AUTHORISED SIGNATORIES" text
            - Often has "A/C PAYEE" stamp diagonally in center
            
            3. PUNJAB NATIONAL BANK:
            - Has "PNB" or "Punjab National Bank" logo
            - Signature appears bottom-right with "Authorised Signatory(ies)" text
            - May include "Proprietor" designation
            
            4. INDIAN OVERSEAS BANK:
            - Has "Indian Overseas Bank" text/logo
            - Signature appears bottom-right with "PARTNER" text
            - Often includes branch details like "KOLKATA-700007"
            
            5. ANY OTHER INDIAN BANK:
            - Signature is almost always in bottom-right quadrant
            - Look for signature line and accompanying text like:
                * "Authorised Signatory"
                * "For [COMPANY]"
                * "Please sign above"
                * "Proprietor" / "Partner" / "Director"
            
            # SIGNATURE EXTRACTION INSTRUCTIONS:
            
            1. THOROUGHLY EXAMINE the entire cheque image
            2. IDENTIFY the signature area (typically bottom-right quadrant)
            3. LOCATE any accompanying text like "Authorised Signatories"
            4. DETERMINE precise normalized coordinates (0.0-1.0 scale)
            5. ADD GENEROUS PADDING (at least 30%) around the signature
            6. INCLUDE all designation text below the signature
            7. If multiple signatures, CAPTURE ALL OF THEM if possible
            
            # THE MOST IMPORTANT RULE:
            If there is ANY doubt about the signature boundaries, ALWAYS BE MORE GENEROUS and include more area rather than less.
            
            Return ONLY this JSON object with NO additional text:
            {
                "exists": true,
                "coordinates": {
                    "x1": 0.65,  /* Left coordinate (0-1) with GENEROUS margin */
                    "y1": 0.55,  /* Top coordinate (0-1) with GENEROUS margin */
                    "x2": 0.98,  /* Right coordinate (0-1) with GENEROUS margin */
                    "y2": 0.85   /* Bottom coordinate (0-1) with GENEROUS margin */
                },
                "description": "Detailed description of signature and its location",
                "confidence": 0.95
            }
            """
            
            # Generate signature coordinates
            signature_response = model.generate_content([
                signature_prompt,
                file_part
            ])
            
            # Extract JSON from response
            signature_json_str = ChequeProcessor._extract_json_from_text(signature_response.text.strip())
            
            try:
                signature_result = json.loads(signature_json_str)
                return signature_result
            except json.JSONDecodeError:
                logger.error(f"JSON parsing error in signature extraction: {signature_json_str[:500]}...")
                return {
                    "exists": False,
                    "coordinates": None,
                    "description": "Failed to extract signature coordinates",
                    "confidence": 0.0
                }
        
        except Exception as e:
            logger.error(f"Error during signature coordinate extraction: {str(e)}")
            return {
                "exists": False,
                "coordinates": None,
                "description": f"Error: {str(e)}",
                "confidence": 0.0
            }

    @staticmethod
    def _extract_json_from_text(text: str) -> str:
        """
        Extract valid JSON from potentially messy text that might contain
        markdown code blocks, explanations, etc.
        """
        # Step 1: Remove markdown code blocks if present
        if "```json" in text:
            # Extract content between ```json and ``` markers
            import re
            json_pattern = r'```json\s*([\s\S]*?)\s*```'
            matches = re.findall(json_pattern, text)
            if matches:
                return matches[0].strip()
        
        # Step 2: If no markdown blocks, try to find the first { and last }
        if '{' in text and '}' in text:
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx < end_idx:
                return text[start_idx:end_idx].strip()
        
        # Step 3: If all else fails, return the input text after removing common non-JSON elements
        clean_text = re.sub(r'^.*?(?=\{)', '', text, flags=re.DOTALL)  # Remove everything before first {
        clean_text = re.sub(r'(?<=\}).*$', '', clean_text, flags=re.DOTALL)  # Remove everything after last }
        
        return clean_text.strip()

    @staticmethod
    def process_multimodal_document(file_data: bytes, file_type: str, file_path: str) -> Dict[str, Any]:
        """Process a cheque document using Vertex AI's multimodal capabilities."""
        try:
            # Initialize Vertex AI model
            model = GenerativeModel("gemini-1.5-pro", safety_settings=safety_settings)
            
            result = {
                "text": "",
                "extracted_fields": [],
                "pages": []
            }
            
            # Process only image files
            if file_type.lower() in ["image/jpeg", "image/jpg", "image/png", "image/tiff"]:
                # Create a Vertex AI Part from the file data
                file_part = Part.from_data(data=file_data, mime_type=file_type)
                
                # Define fields and their descriptions
                doc_fields = [field['name'] for field in FIELDS]
                fields_str = ", ".join(doc_fields)

                field_descriptions = {
                    "bank_name": (
                        "**Primary Task:** Extract the official name of the bank.\n"
                        "**Location:** Prioritize the name prominently displayed in the top-left quadrant of the cheque.\n"
                        "**Disambiguation:** If multiple bank names appear, strictly select the one in the top-left corner.\n"
                        "**Method:** Use visual layout cues, contextual knowledge of major Indian and international bank names/logos, and pattern recognition. Cross-reference with IFSC/MICR if needed.\n"
                        "**Challenge:** Faded print, stylized logos."
                    ),
                    "bank_branch": (
                        "**Primary Task:** Identify the specific branch name or location.\n"
                        "**Location:** Typically found directly below or adjacent to the `bank_name`.\n"
                        "**Content:** May include branch name, city, area code, or a combination.\n"
                        "**Method:** Look for text indicating a location or branch designation near the bank name.\n"
                        "**Challenge:** Can be less prominent than the bank name, might be abbreviated."
                    ),
                    "account_number": (
                        "**Primary Task:** Extract the unique identifier of the customer's bank account.\n"
                        "**Location:** Variable, often near bank details, payee line, or MICR strip. Look for labels like 'A/C No.', 'Account No.', 'SB Acct', etc.\n"
                        "**Format:** Primarily numeric, but may contain hyphens or spaces (ignore separators in final value). Can occasionally be alphanumeric in some banks.\n"
                        "**Method:** Identify sequences of digits (typically 9-16 digits) associated with account labels. Apply rigorous character differentiation (0/O, 1/I, 5/S, 8/B).\n"
                        "**Challenge:** OCR errors on digits, varying lengths, potential prefixes/suffixes."
                    ),
                    "date": (
                        "**Primary Task:** Extract the issue date of the cheque and standardize it.\n"
                        "**Location:** Almost always in the top-right corner, often within designated boxes (DD MM YYYY).\n"
                        "**Input Format:** Handle various formats (DDMMYYYY, DD/MM/YYYY, DD-MM-YY, DD.MM.YYYY, DD Mon YYYY, etc.), including handwritten variations and partial pre-fills (e.g., printed '20__').\n"
                        "**Output Format:** **Strictly YYYY-MM-DD.**\n"
                        "**Method:** Parse day, month, and year components. Use context (assume DD/MM for India if ambiguous) and validate as a real calendar date within a reasonable timeframe (e.g., past 6 months).\n"
                        "**Challenge:** Illegible handwriting, ambiguous single digits, strikethroughs/corrections (extract the final date)."
                    ),
                    "payee_name": (
                        "**Primary Task:** Extract the name of the recipient (person or entity).\n"
                        "**Location:** Follows the keyword 'PAY', 'Pay To', 'Pay to the order of', usually below the bank details.\n"
                        "**Content:** Typically handwritten, can span multiple lines, may include titles (Mr./Ms./M/s).\n"
                        "**Method:** Focus on the text following 'PAY'. Apply advanced handwriting recognition.\n"
                        "**Challenge:** Highly variable handwriting, legibility issues, potential for non-English names/scripts (Hindi, Tamil, etc.), overlapping text."
                    ),
                    "amount_words": (
                        "**Primary Task:** Extract the cheque amount written out in words (legal amount).\n"
                        "**Location:** Usually below the `payee_name`, often starting with 'Rupees' or the currency name, may span two lines, often ends with 'Only'.\n"
                        "**Content:** Primarily handwritten text representing the numeric value.\n"
                        "**Method:** Apply advanced handwriting recognition. Capture all relevant text, including currency units (e.g., 'Rupees', 'Dollars') and fractional units ('Paise', 'Cents') if present. Cross-validate logic with `amount_numeric`.\n"
                        "**Challenge:** Difficult handwriting, non-English words/scripts (e.g., 'हजार', 'லட்சம்'), abbreviations, spanning multiple lines."
                    ),
                    "amount_numeric": (
                        "**Primary Task:** Extract the cheque amount written in figures.\n"
                        "**Location:** Typically in a designated box or area on the right-middle side of the cheque, often preceded or followed by a currency symbol (₹, $, €) or code.\n"
                        "**Format:** Numeric digits, potentially with commas as thousands separators and a period/dot as the decimal separator. May end with '/-' or '.00'.\n"
                        "**Method:** Identify digits within the amount box. Capture the full value including decimals. Note adjacent currency symbol for `currency` field determination. Apply rigorous digit OCR. Cross-validate with `amount_words`.\n"
                        "**Challenge:** Handwritten digits clarity (esp. 1/7, 4/9, 2/5), overlapping currency symbols, corrections."
                    ),
                    "currency": (
                        "**Primary Task:** Identify the currency of the transaction.\n"
                        "**Method:** Infer from: \n    1. Explicit currency codes (e.g., INR, USD, EUR) printed on the cheque.\n    2. Currency symbols next to `amount_numeric` (e.g., ₹, $, £, €).\n    3. Currency words in `amount_words` (e.g., 'Rupees', 'Dollars').\n    4. Default to context (e.g., assume 'INR' for identified Indian banks if no other indicators).\n"
                        "**Output:** Standard 3-letter ISO code (e.g., INR, USD, EUR).\n"
                        "**Challenge:** Ambiguity if multiple indicators conflict or none are present."
                    ),
                    "issuer_name": (
                        "**Primary Task:** Extract the name of the account holder issuing the cheque.\n"
                        "**Location:** Typically appears below the signature space or sometimes above it, usually on the bottom-right, above the MICR line.\n"
                        "**Content:** Can be printed or handwritten; represents the individual or company name associated with the account.\n"
                        "**Method:** Look for names (personal or company) in the designated area near the signature. Differentiate from payee name.\n"
                        "**Challenge:** Might be missing, illegible if handwritten, or just a company stamp."
                    ),
                    "micr_code": (
                        "**Primary Task:** Extract the Magnetic Ink Character Recognition code.\n"
                        "**Location:** Distinctive E-13B font printed along the bottom edge of the cheque.\n"
                        "**Format:** **DIGITS ONLY (0-9)**. Exclude any special surrounding symbols (like ⑆, ⑈, ⑇).\n"
                        "**Structure (India):** Typically 9 digits (e.g., CityCode-BankCode-BranchCode: 3-3-3).\n"
                        "**Method:** Target the specific font and location. Apply OCR trained for E-13B. Validate digit-only content and expected length.\n"
                        "**Challenge:** Faint/broken print, ink smudges, non-standard characters mistakenly included."
                    ),
                    "signature_present": (
                        "**Primary Task:** Determine if a handwritten signature exists.\n"
                        "**Location:** Assess the designated signature area, usually bottom-right, above the `issuer_name` (if present) and MICR line.\n"
                        "**Method:** Detect the presence of free-flowing, handwritten ink strokes consistent with a signature. Do *not* attempt to read or validate the signature itself.\n"
                        "**Output:** **Strictly YES or NO.**\n"
                        "**Challenge:** Faint signatures, stamps overlapping the area, distinguishing stray marks from an intended signature."
                    ),
                    "IFSC": (
                        "**Primary Task:** Extract the Indian Financial System Code.\n"
                        "**Location:** Usually printed near the bank name/branch details, often explicitly labeled 'IFSC', 'IFS Code', or similar.\n"
                        "**Format:** Alphanumeric, **strictly 11 characters** long. The first 4 are alphabetic (bank code), the 5th is always '0', and the last 6 are alphanumeric (branch code). Example: SBIN0001234.\n"
                        "**Method:** Look for the 11-character code matching the format, often near the bank details. Validate format rigorously. Cross-reference first 4 chars with `bank_name`.\n"
                        "**Challenge:** Small print size, potential OCR errors confusing letters/numbers (0/O, 1/I, S/5, B/8)."
                    )
                }

                # Create field list with descriptions
                fields_with_descriptions = []
                for field in doc_fields:
                    description = field_descriptions.get(field, "")
                    fields_with_descriptions.append(f"- {field}: {description}")
                
                fields_list = "\n".join(fields_with_descriptions)
  
                extraction_prompt = f"""You are an expert AI assistant specializing in high-accuracy information extraction from scanned cheque images, leveraging advanced multimodal understanding, sophisticated OCR, and deep domain knowledge of global banking practices, particularly Indian cheques. Your task is to meticulously analyze the provided text representation of a cheque and extract specific fields with maximum precision and confidence.

                Assume the input text originates from a high-resolution scan, but be prepared to handle potential OCR errors, variations in image quality (blur, low contrast, skew), handwriting ambiguities, and multilingual content.

                **Core Objective:** Extract the specified fields from the cheque data.

                **Field Definitions & Extraction Guidelines:**

                {fields_list}

                **Critical Extraction Principles & Guidelines:**

                1.  **Contextual Reasoning:** Apply deep contextual understanding. Use knowledge of cheque layouts, banking terminology (Indian and international), common payee names, and standard formats to interpret information correctly. Cross-validate information between fields (e.g., amount words vs. numeric amount, bank name vs. IFSC/MICR).
                2.  **Character Differentiation (Precision Focus):**
                    *   Actively disambiguate visually similar characters (0/O, 1/I/l, 2/Z, 5/S, 8/B, ./,, :/; etc.). Pay extreme attention in critical fields like Account Numbers, MICR, IFSC, and Amounts.
                    *   Recognize common OCR ligatures/errors (rn/m, cl/d, vv/w) and correct them based on context.
                    *   Verify character types against field expectations (e.g., digits in `account_number`, `amount_numeric`, `micr_code`, `IFSC`; predominantly letters in names).
                3.  **Advanced Handwriting Analysis:**
                    *   Employ sophisticated handwriting recognition models capable of handling diverse styles (cursive, print, mixed), varying slant, inconsistent spacing/size, loops, pressure points, and potential overlaps or incompleteness.
                    *   Specifically address challenges in handwritten: `payee_name`, `amount_words`, `amount_numeric`, `date`, `issuer_name`, and `signature_present` assessment.
                    *   Accurately interpret handwritten numbers, distinguishing styles for '1'/'7', '4'/'9', '2', etc., even when connected.
                    *   Handle corrections (strikethroughs): Prioritize the final, intended value, not the crossed-out text. If a date is corrected, extract the corrected date.
                4.  **Multilingual & Mixed-Script Processing:**
                    *   Accurately identify and transcribe text in multiple languages, primarily English and major Indian languages (Hindi, Kannada, Telugu, Tamil, Punjabi, Bengali, etc.).
                    *   Specify the detected language for fields prone to multilingual content (`payee_name`, `amount_words`, `issuer_name`) if not English.
                    *   Apply script-specific character differentiation rules (e.g., Devanagari ण/ज़, த/த; Tamil ன/ண, ர/ற; similar forms in Telugu/Kannada/Bengali/Assamese).
                    *   Handle code-switching (mixing scripts/languages) within a single field value where appropriate.
                    *   Recognize and correctly transcribe Indian language numerals if present.
                5.  **MICR Code Extraction:**
                    *   Target the E-13B font sequence at the cheque bottom.
                    *   Extract **digits only (0-9)**. Explicitly **exclude** any non-digit symbols or delimiters (like ⑆, ⑈, ⑇).
                    *   Validate the typical 9-digit structure for Indian cheques (CCCBBBAAA - City, Bank, Branch). Note variations if necessary.
                    *   Ensure high confidence differentiation of MICR's unique blocky characters.
                6.  **Date Extraction & Standardization:**
                    *   Locate the date, typically top-right.
                    *   Recognize various formats (DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD, DD-Mon-YYYY, etc.) including handwritten variations.
                    *   Handle partial pre-fills (e.g., printed "20" followed by handwritten "24").
                    *   Accurately parse day, month, and year, resolving ambiguity using context (assume DD/MM for India unless clearly otherwise) and proximity to the likely processing date (cheques are typically valid for 3-6 months).
                    *   Standardize the final output strictly to **YYYY-MM-DD** format. If the date is invalid or ambiguous (e.g., Feb 30), flag it.
                7.  **Amount Validation:** Ensure `amount_numeric` and `amount_words` correspond logically. Note discrepancies if unavoidable. Extract numeric amount precisely, including decimals if present.
                8.  **Signature Detection:** Assess the presence of handwritten, free-flowing ink strokes in the typical signature area (bottom right, above MICR). Output only "YES" or "NO". Do not attempt to read the signature text itself for the `signature_present` field.

                **Confidence Scoring:**

                *   Assign a confidence score (0.0 to 1.0) for each extracted field based on clarity, ambiguity, potential OCR issues, handwriting legibility, and adherence to expected formats.
                *   Use these benchmarks:
                    *   0.95-1.00: High confidence, clear source, unambiguous.
                    *   0.85-0.94: Very high confidence, minor potential ambiguity resolved by context.
                    *   0.75-0.84: Good confidence, some ambiguity or minor OCR/handwriting difficulty.
                    *   0.65-0.74: Moderate confidence, noticeable issues (e.g., slightly blurry text, difficult handwriting).
                    *   0.50-0.64: Low confidence, significant recognition problems, extraction is uncertain.
                    *   <0.50: Very low confidence, likely incorrect or speculative.
                *   **Confidence Justification:** Briefly explain scores below 0.9 (e.g., 'Moderate confidence due to cursive handwriting in payee name', 'Low confidence due to faint printing of MICR').
                *   **Handwriting Confidence Modifiers:** Adjust scores based on handwriting quality: slightly lower for complex cursive or corrections, higher for clear block printing.

                **Error Handling:**

                *   If a field cannot be found or reliably extracted, set its value to `null` or an empty string, assign a low confidence score (e.g., < 0.5), and provide a specific `reason` (e.g., "Field not present", "Illegible handwriting", "Smudged area", "OCR segmentation failed").

                **Output Format:**

                *   Your response **MUST** be a single, valid JSON object.
                *   **Do NOT** include any explanatory text, markdown formatting, or anything outside the JSON structure.
                *   The JSON should have two top-level keys:
                    1.  `"full_text"`: A string containing the entire OCR text extracted from the cheque, as accurately as possible.
                    2.  `"extracted_fields"`: An array of objects. Each object represents an extracted field and must contain:
                        *   `"field_name"`: The name of the field (string, e.g., "bank_name").
                        *   `"value"`: The extracted value (string, number, or boolean for `signature_present`). Standardize date to "YYYY-MM-DD". Null or "" if not found/extractable.
                        *   `"confidence"`: The confidence score (float, 0.0-1.0).
                        *   `"text_segment"`: The exact text substring from the source OCR corresponding to the extracted value (string). Null if not applicable.
                        *   `"reason"`: A brief reason if the field could not be extracted or confidence is low (string). Null or empty otherwise.
                        *   `"language"`: (Optional, but preferred for `payee_name`, `amount_words`, `issuer_name`) The detected language of the extracted value (string, e.g., "English", "Hindi", "Tamil"). Null if not applicable or detection failed.


                **Example extracted_fields object will contain all these fields with example values like:
                    "field_name": "amount_numeric",
                    "value": "1500.00",
                    "confidence": 0.98,
                    "text_segment": "1500/-",
                    "reason": null,
                    "language": "English"
                    
                IMPORTANT: Your response must be a valid JSON object and NOTHING ELSE. No explanations, no markdown code blocks.
                """  

                extraction_response = model.generate_content([
                    extraction_prompt,
                    file_part
                ])
                
                extraction_json_str = extraction_response.text.strip()
                logger.info(f"Raw extraction response length: {len(extraction_json_str)}")
                logger.info(f"First 500 chars of extraction response: {extraction_json_str}...")
                
                # Clean up the JSON string to handle markdown code blocks and any text before/after
                extraction_json_str = ChequeProcessor._extract_json_from_text(extraction_json_str)
                
                try:
                    extraction_result = json.loads(extraction_json_str)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error in extraction: {e}, Raw response preview: {extraction_json_str[:500]}...")
                    # Try harder to extract valid JSON
                    try:
                        import re
                        json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
                        match = re.search(json_pattern, extraction_json_str)
                        if match:
                            potential_json = match.group(0)
                            extraction_result = json.loads(potential_json)
                            logger.info("Successfully extracted JSON using regex pattern")
                        else:
                            raise ValueError("Could not find valid JSON pattern")
                    except Exception as inner_e:
                        logger.error(f"Advanced JSON extraction also failed: {inner_e}")
                        # Provide fallback extraction result
                        extraction_result = {
                            "full_text": "Failed to extract text due to JSON parsing error",
                            "extracted_fields": []
                        }
                
                # Combine the results
                result = {
                    "text": extraction_result.get("full_text", ""),
                    "extracted_fields": extraction_result.get("extracted_fields", []),
                    "pages": [{"page_num": 1, "text": extraction_result.get("full_text", "")}]
                }
                
                # Extract signature coordinates
                signature_result = ChequeProcessor.extract_signature_coordinates(file_data, file_type)
                
                # Add signature coordinates to the result
                result['signature_coordinates'] = signature_result
                
                # If signature exists, add it to extracted fields
                if signature_result.get('exists', False):
                    result['extracted_fields'].append({
                        "field_name": "signature_coordinates",
                        "value": json.dumps(signature_result['coordinates']) if signature_result['coordinates'] else "N/A",
                        "confidence": signature_result.get('confidence', 0.0),
                        "reason": signature_result.get('description', '')
                    })
                
            else:
                logger.warning(f"Unsupported file type for Vertex AI processing: {file_type}")
                result = {
                    "error": f"Unsupported file type: {file_type}",
                    "text": "",
                    "pages": [],
                    "extracted_fields": [],
                    "signature_coordinates": {
                        "exists": False,
                        "coordinates": None,
                        "description": "Unsupported file type",
                        "confidence": 0.0
                    }
                }
                
            return result
            
        except Exception as e:
            logger.error(f"Error during multimodal document processing: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "text": "",
                "pages": [],
                "document_type": "unknown",
                "confidence": 0.0,
                "extracted_fields": [],
                "signature_coordinates": {
                    "exists": False,
                    "coordinates": None,
                    "description": str(e),
                    "confidence": 0.0
                }
            }

    @staticmethod
    def process_document_batch(file_batch):
        """
        Process a batch of documents in parallel using ThreadPoolExecutor
        
        file_batch: list of dictionaries with keys 'data', 'type', and 'path'
        """
        results = []
        
        # Create a ThreadPoolExecutor for parallel processing within the batch
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(file_batch), MAX_WORKERS)) as batch_executor:
            # Submit all files in this batch for processing
            futures = {
                batch_executor.submit(ChequeProcessor.process_multimodal_document, 
                                     file_info['data'], 
                                     file_info['type'],
                                     file_info['path']): file_info 
                for file_info in file_batch
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                file_info = futures[future]
                try:
                    result = future.result()
                    # Add file path to result for tracking
                    result['file_path'] = file_info['path']
                    results.append(result)
                    logger.info(f"Successfully processed {file_info['path']}")
                except Exception as e:
                    logger.error(f"Error processing file {file_info['path']}: {str(e)}")
                    # Add error result
                    results.append({
                        "error": str(e),
                        "file_path": file_info['path'],
                        "extracted_fields": []
                    })
        
        return results
    
def crop_image_with_coordinates(image_path, coordinates):
    """
    Crop an image based on normalized coordinates with extra padding.
    
    Args:
        image_path (str): Path to the original image
        coordinates (dict): Normalized coordinates (x1, y1, x2, y2)
    
    Returns:
        bytes: Cropped image as bytes
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Get image dimensions
            width, height = img.size
            
            # Extract coordinates with generous padding
            x1 = int(coordinates.get("x1", 0.65) * width)
            y1 = int(coordinates.get("y1", 0.60) * height)
            x2 = int(coordinates.get("x2", 0.98) * width)
            y2 = int(coordinates.get("y2", 0.90) * height)
            
            # Add very generous padding (30%) to ensure we don't miss parts of the signature
            # For Indian cheques where the signature area can include designations and "Please sign above" text
            padding_x = int((x2 - x1) * 0.30)
            padding_y = int((y2 - y1) * 0.30)
            
            # Apply padding while ensuring we stay within image bounds
            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y)
            x2 = min(width, x2 + padding_x)
            y2 = min(height, y2 + padding_y)
            
            logger.info(f"Final signature crop coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # Crop image without resizing
            cropped_img = img.crop((x1, y1, x2, y2))
            
            # Save to bytes
            img_byte_arr = io.BytesIO()
            cropped_img.save(img_byte_arr, format='PNG')
            return img_byte_arr.getvalue()
    except Exception as e:
        logger.error(f"Error cropping image: {e}")
        return None
        
def process_zip_files(file_contents: List[bytes], file_names: List[str], job_id: str):
    """
    Process multiple zip files and generate Excel report with signature image cropping.
    
    Args:
        file_contents (List[bytes]): List of zip file contents
        file_names (List[str]): List of corresponding zip file names
        job_id (str): Unique identifier for the processing job
    
    Returns:
        str: Path to the generated Excel output file
    """
    # Initialize logging and tracking variables
    logger.info(f"Starting optimized process_zip_files for job {job_id}")
    logger.info(f"Number of files: {len(file_contents)}")
    logger.info(f"File names: {file_names}")

    # Job start time and file tracking
    job_start_time = time.time()
    total_files = 0
    processed_files = 0

    try:
        # Create temporary directories for processing
        temp_dir = tempfile.mkdtemp(prefix=f"job_{job_id}_")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a directory for temporary signature images
        signature_dir = os.path.join(output_dir, "signatures")
        os.makedirs(signature_dir, exist_ok=True)
        
        # Dictionary to store results for each folder
        folder_results = {}
        
        # Process each uploaded zip file
        for zip_index, (zip_content, zip_name) in enumerate(zip(file_contents, file_names)):
            # Generate unique directory for this zip file
            zip_dir = os.path.join(temp_dir, f"zip_{zip_index}_{os.path.splitext(zip_name)[0]}")
            os.makedirs(zip_dir, exist_ok=True)
            
            # Extract zip contents
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                zf.extractall(zip_dir)
            
            # Prepare file collection for batch processing
            folder_files = {}
            
            # Traverse directory to find processable files
            for root, dirs, files in os.walk(zip_dir):
                # Skip root zip directory
                if root == zip_dir:
                    continue
                
                # Get relative path for folder naming
                rel_path = os.path.relpath(root, zip_dir)
                folder_name = rel_path if rel_path != '.' else zip_name
                
                # Skip empty directories
                if not files:
                    continue
                
                # Initialize folder results and files lists
                if folder_name not in folder_results:
                    folder_results[folder_name] = []
                
                if folder_name not in folder_files:
                    folder_files[folder_name] = []
                
                # Process files in the directory
                for file in files:
                    # Skip hidden or temporary files
                    if file.startswith('.') or file.startswith('~'):
                        continue
                    
                    file_path = os.path.join(root, file)
                    total_files += 1
                    
                    # Get file extension
                    _, ext = os.path.splitext(file)
                    
                    # Support only specific image and document types
                    supported_extensions = {
                        '.pdf': 'application/pdf',
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                        '.tiff': 'image/tiff',
                        '.tif': 'image/tiff'
                    }
                    
                    # Check if file type is supported
                    if ext.lower() not in supported_extensions:
                        logger.warning(f"Unsupported file type: {file_path}")
                        continue
                    
                    # Determine file MIME type
                    file_type = supported_extensions[ext.lower()]
                    
                    # Read file data
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                    
                    # Add to folder files for batch processing
                    folder_files[folder_name].append({
                        'path': file_path,
                        'data': file_data,
                        'type': file_type
                    })
            
            # Process files in batches for each folder
            for folder_name, files_list in folder_files.items():
                logger.info(f"Processing folder {folder_name} with {len(files_list)} files")
                
                # Process files in batches
                for i in range(0, len(files_list), BATCH_SIZE):
                    batch = files_list[i:i+BATCH_SIZE]
                    logger.info(f"Processing batch {i//BATCH_SIZE + 1} with {len(batch)} files")
                    
                    # Process batch in parallel
                    batch_results = ChequeProcessor.process_document_batch(batch)
                    
                    # Process and collect batch results
                    for result in batch_results:
                        file_path = result.get('file_path', '')
                        
                        # Extract and store fields
                        for field in result.get("extracted_fields", []):
                            field_entry = {
                                "filepath": file_path,
                                "field_name": field.get("field_name", ""),
                                "value": field.get("value", ""),
                                "confidence": field.get("confidence", 0.0),
                                "reason": field.get("reason", "")
                            }
                            
                            # Special handling for signature coordinates
                            if field.get("field_name") == "signature_coordinates":
                                try:
                                    # Parse coordinates
                                    coords = json.loads(field.get("value", "{}"))
                                    
                                    # Crop signature image
                                    signature_img = crop_image_with_coordinates(
                                        file_path, 
                                        coords
                                    )
                                    
                                    if signature_img:
                                        # Generate unique filename for signature
                                        signature_filename = f"signature_{processed_files}_{uuid.uuid4()}.png"
                                        signature_path = os.path.join(signature_dir, signature_filename)
                                        
                                        # Save signature image
                                        with open(signature_path, 'wb') as sig_file:
                                            sig_file.write(signature_img)
                                        
                                        # Add signature image path to field entry
                                        field_entry["signature_image_path"] = signature_path
                                except Exception as e:
                                    logger.error(f"Error processing signature for {file_path}: {e}")
                            
                            folder_results[folder_name].append(field_entry)
                        
                        processed_files += 1
                        
                        # Periodic progress update
                        if processed_files % 10 == 0:
                            current_time = time.time()
                            elapsed_time = current_time - job_start_time
                            
                            # Calculate processing rate and estimated time remaining
                            if processed_files > 0:
                                files_per_second = processed_files / elapsed_time
                                remaining_files = total_files - processed_files
                                estimated_time_remaining = (remaining_files / files_per_second) if files_per_second > 0 else 0
                                
                                # Format time remaining
                                hours, remainder = divmod(estimated_time_remaining, 3600)
                                minutes, seconds = divmod(remainder, 60)
                                time_format = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                                
                                # Log progress
                                logger.info(
                                    f"Progress: {processed_files}/{total_files} files processed "
                                    f"({processed_files/total_files*100:.1f}%). "
                                    f"Processing rate: {files_per_second:.2f} files/sec. "
                                    f"Estimated time remaining: {time_format}"
                                )
        
        # Generate comprehensive Excel report
        excel_path = os.path.join(output_dir, f"cheque_extraction_results_{job_id}.xlsx")
        
        # Create Excel writer with xlsxwriter engine to support images
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            # Process results for each folder
            for folder_name, results in folder_results.items():
                if not results:
                    continue
                
                # Prepare data for DataFrame
                filepath_groups = {}
                signature_images = {}
                
                for item in results:
                    filepath = item["filepath"]
                    if filepath not in filepath_groups:
                        filepath_groups[filepath] = {"filepath": filepath}
                    
                    # Add field details
                    if "field_name" in item:
                        field_name = item["field_name"]
                        
                        # Store signature image path if available
                        if "signature_image_path" in item:
                            signature_images[filepath] = item["signature_image_path"]
                        
                        filepath_groups[filepath][field_name] = item["value"]
                        filepath_groups[filepath][f"{field_name}_conf"] = item["confidence"]
                        
                        if item.get("reason"):
                            filepath_groups[filepath][f"{field_name}_reason"] = item["reason"]
                
                # Create DataFrame
                if filepath_groups:
                    df = pd.DataFrame(list(filepath_groups.values()))
                    
                    # Define column order
                    cols = ["filepath"]
                    for field in FIELDS + [{"name": "signature_coordinates"}]:
                        field_name = field["name"]
                        if field_name in df.columns:
                            cols.append(field_name)
                            cols.append(f"{field_name}_conf")
                            if f"{field_name}_reason" in df.columns:
                                cols.append(f"{field_name}_reason")
                    
                    # Select existing columns
                    cols = [col for col in cols if col in df.columns]
                    if cols:
                        df = df[cols]
                    
                    # Sanitize sheet name
                    sheet_name = re.sub(r'[\\/*?[\]:]', '_', folder_name)
                    sheet_name = (sheet_name[:28] + '...') if len(sheet_name) > 31 else sheet_name
                    
                    # Write to Excel with image support
                    if not df.empty:
                        # Write DataFrame to Excel
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # Get the xlsxwriter workbook and worksheet objects
                        workbook = writer.book
                        worksheet = writer.sheets[sheet_name]
                        
                        # Add image column for signatures
                        signature_img_col = len(cols)  # Add new column after existing columns
                        
                        # Add signature images
                        for idx, row in df.iterrows():
                            filepath = row['filepath']
                            
                            # Check if signature image exists for this filepath
                            if filepath in signature_images:
                                try:
                                    img_path = signature_images[filepath]
                                    
                                    # Insert image into worksheet
                                    worksheet.insert_image(
                                        idx + 1,  # Excel rows are 1-indexed, and first row is header
                                        signature_img_col, 
                                        img_path,
                                        {'x_scale': 0.5, 'y_scale': 0.5}  # Adjust scaling as needed
                                    )
                                except Exception as e:
                                    logger.error(f"Error inserting signature image for {filepath}: {e}")
        
        # Update job status
        job_end_time = time.time()
        processed_jobs[job_id] = {
            "status": "completed",
            "start_time": job_start_time,
            "end_time": job_end_time,
            "total_files": total_files,
            "processed_files": processed_files,
            "output_file_path": excel_path,
            "processing_time": job_end_time - job_start_time,
            "signature_directory": signature_dir  # Keep track of generated signature images
        }
        
        logger.info(f"Job {job_id} completed. Output file: {excel_path}")
        return excel_path
    
    except Exception as e:
        # Comprehensive error handling
        logger.error(f"Error processing zip files: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Update job status to failed
        processed_jobs[job_id] = {
            "status": "failed",
            "start_time": job_start_time,
            "end_time": time.time(),
            "total_files": total_files,
            "processed_files": processed_files,
            "error_message": str(e),
            "error_traceback": traceback.format_exc()
        }
        
        # Raise the exception to be handled by the caller
        raise
    
@app.get("/download/{job_id}")
async def download_results(job_id: str):
    """Download the results of a completed job."""
    try:
        # Fetch job from processed_jobs
        job = processed_jobs.get(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
        
        if job.get("status") != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Job is not completed. Current status: {job.get('status', 'unknown')}"
            )
        
        output_path = job.get("output_file_path")
        
        if not output_path or not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail="Output file not found")
        
        return FileResponse(
            path=output_path,
            filename=f"cheque_extraction_results_{job_id}.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Add a new endpoint to check job status
@app.get("/status/{job_id}")
async def check_job_status(job_id: str):
    """Check the status of a processing job."""
    try:
        # Check if the job is still running
        if job_id in active_tasks:
            future = active_tasks[job_id]
            if future.done():
                if future.exception():
                    return {
                        "status": "failed",
                        "error": str(future.exception())
                    }
                else:
                    return {
                        "status": "completed",
                        "result": "Processing complete. Use the /download endpoint to get results."
                    }
            else:
                return {
                    "status": "processing",
                    "message": "Job is still being processed."
                }
        
        # Check in processed jobs
        if job_id in processed_jobs:
            job = processed_jobs[job_id]
            return {
                "status": job.get("status", "unknown"),
                "total_files": job.get("total_files", 0),
                "processed_files": job.get("processed_files", 0),
                "start_time": job.get("start_time"),
                "end_time": job.get("end_time", None),
                "error_message": job.get("error_message", None)
            }
        
        # Job not found
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...)
):
    """
    Upload zip files containing folders of cheque images.
    
    Args:
        files: List of zip files to be processed
    
    Returns:
        Dict containing job status, job ID, and initial processing information
    """
    try:
        # Validate input files
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        # Validate file types (ensure they are zip files)
        for file in files:
            if not file.filename.lower().endswith('.zip'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid file type for {file.filename}. Only ZIP files are accepted."
                )
        
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Define a wrapper function to handle exceptions and update the job status
        def process_wrapper():
            try:
                # Read the file contents into memory before processing
                file_contents = []
                file_names = []
                for file in files:
                    content = file.file.read()
                    file_contents.append(content)
                    file_names.append(file.filename)
                
                # Initialize job status in-memory with comprehensive tracking
                processed_jobs[job_id] = {
                    "status": "processing",
                    "start_time": time.time(),
                    "total_files": 0,
                    "processed_files": 0,
                    "job_id": job_id,
                    "input_files": file_names,
                    "progress_percentage": 0.0,
                    "estimated_time_remaining": None
                }
                
                # Process the files
                output_file = process_zip_files(file_contents, file_names, job_id)
                
                # Update final job status
                processed_jobs[job_id].update({
                    "status": "completed",
                    "end_time": time.time(),
                    "output_file_path": output_file,
                    "progress_percentage": 100.0,
                    "estimated_time_remaining": 0
                })
                
                logger.info(f"Job {job_id} completed successfully")
                
            except Exception as e:
                # Comprehensive error logging
                logger.error(f"Error processing job {job_id}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Update job status to failed with detailed error information
                processed_jobs[job_id] = {
                    "status": "failed",
                    "start_time": processed_jobs[job_id].get('start_time', time.time()),
                    "end_time": time.time(),
                    "job_id": job_id,
                    "error_message": str(e),
                    "error_traceback": traceback.format_exc(),
                    "input_files": file_names,
                    "progress_percentage": 0.0
                }
        
        # Submit task to thread pool executor
        future = executor.submit(process_wrapper)
        
        # Add a callback to handle completion
        def on_complete(future):
            try:
                # Remove the task from active tasks
                active_tasks.pop(job_id, None)
                # If the task completed successfully, the result is already handled in process_zip_files
                if future.exception():
                    logger.error(f"Task for job {job_id} failed: {future.exception()}")
            except Exception as e:
                logger.error(f"Error in on_complete callback: {str(e)}")
        
        future.add_done_callback(on_complete)
        
        # Store the future for tracking
        active_tasks[job_id] = future
        
        return {
            "status": "processing",
            "job_id": job_id,
            "message": "Cheque extraction job initiated successfully",
            "files": [file.filename for file in files],
            "timestamp": int(time.time())
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions directly
        raise
    
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected error in upload endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during file upload: {str(e)}"
        )

if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)