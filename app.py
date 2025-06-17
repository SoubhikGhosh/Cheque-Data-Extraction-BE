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
from typing import List, Dict, Any, Optional
import pandas as pd
import uuid
import shutil
import json
import re
import uvicorn
import concurrent.futures
import traceback
from google.api_core import exceptions as google_exceptions
from datetime import datetime


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
    description="API for processing zip files containing cheque images to extract date and amount.",
    version="2.6.1" # Version updated to reflect prompt restoration
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

# Field definitions for cheques - Simplified to only date and amount
FIELDS = [
    {"id": 1, "name": "date"},
    {"id": 2, "name": "amount"}
]

# ============ PERFORMANCE OPTIMIZATION CONSTANTS ============
MAX_WORKERS = 120
BATCH_SIZE = 40
REASK_CONFIDENCE_THRESHOLD = 0.9 # Trigger reAsk if confidence is below this score

# Create a thread pool executor at the module level
executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Keep a dictionary of futures for tracking
active_tasks = {}
processed_jobs = {}

class ChequeProcessor:
    """Helper class for cheque processing operations using Vertex AI's multimodal capabilities"""

    @staticmethod
    def _call_vertex_ai_with_retry(
        model_instance: GenerativeModel,
        prompt_parts: List[Any],
        max_retries: int = 100,
        initial_delay: float = 1.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ) -> Any:
        """
        Calls the Vertex AI model's generate_content method with exponential backoff.
        """
        num_retries = 0
        delay = initial_delay
        retryable_errors = (
            google_exceptions.ResourceExhausted,
            google_exceptions.TooManyRequests,
            google_exceptions.ServiceUnavailable,
            google_exceptions.DeadlineExceeded
        )

        while True:
            try:
                logger.debug(f"Attempting Vertex AI API call (Attempt {num_retries + 1}/{max_retries + 1})")
                response = model_instance.generate_content(prompt_parts)
                logger.debug(f"Vertex AI API call successful (Attempt {num_retries + 1}/{max_retries + 1})")
                return response
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
                    actual_delay += (hash(str(time.time())) % 100 / 400.0) * delay 

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

    @staticmethod
    def _extract_json_from_text(text: str) -> str:
        """
        Extract valid JSON from potentially messy text that might contain markdown.
        """
        if "```json" in text:
            json_pattern = r'```json\s*([\s\S]*?)\s*```'
            matches = re.findall(json_pattern, text)
            if matches:
                return matches[0].strip()
        
        if '{' in text and '}' in text:
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx < end_idx:
                return text[start_idx:end_idx].strip()
        
        clean_text = re.sub(r'^.*?(?=\{)', '', text, flags=re.DOTALL)
        clean_text = re.sub(r'(?<=\}).*$', '', clean_text, flags=re.DOTALL)
        
        return clean_text.strip()
    
    @staticmethod
    def process_multimodal_document(file_data: bytes, file_type: str, file_path: str) -> Dict[str, Any]:
        """Process a cheque document using Vertex AI with a reAsk strategy for failed extractions."""
        try:
            model = GenerativeModel("gemini-1.5-flash-002", safety_settings=safety_settings)
            file_part = Part.from_data(data=file_data, mime_type=file_type)

            # =========== DYNAMIC YEAR CALCULATION ===========
            current_year = datetime.now().year
            previous_year = current_year - 1

            # =========== INITIAL PROMPTS (FIRST PASS) ===========
            # This dictionary is now defined inside the method to use the dynamic years
            field_descriptions = {
                "date": (
                "**Objective:** You are a hyper-precise OCR engine. Your task is to extract the 8-digit date from the provided pre-cropped image of a cheque's date grid."
                    "\n\n"
                    "**CRITICAL INTERNAL PROCESS (Follow these steps before giving your answer):**"
                    "\n"
                    "1.  **Digit-by-Digit Analysis:** Mentally scan each of the 8 boxes. Note the digit in each box, or make an educated guess based on context if it is unclear. The primary challenge is to ignore the printed box lines and focus only on the handwritten ink."
                    "\n"
                    "2.  **Correction Check:** Look for any strikethroughs (horizontal or vertical). If a corrected date is present, you must use the new, valid date."
                    "\n"
                    "3.  **Apply Temporal Rule:** The current year is **{current_year}**. A valid cheque date will almost certainly be for the year **{current_year}** or late **{previous_year}**. \n"
                    f"    * An extracted year like '{previous_year - 1}' or '{current_year + 1}' is extremely improbable. Use this rule to disambiguate OCR errors. For example, if the last digit of the year is ambiguous between a '{str(current_year)[-1]}' and a '{str(current_year + 1)[-1]}', you must conclude it is '{str(current_year)[-1]}' to form '{current_year}', as '{current_year + 1}' is not a plausible cheque date. A low confidence score must be assigned if the only possible reading is an invalid year.\n"
                    "\n"
                    "4.  **Assemble and Validate:** Combine the recognized digits. If the result is not a valid calendar date (e.g., '31-04-2025'), it is invalid."
                    "\n\n"
                    "**FINAL OUTPUT INSTRUCTION:**"
                    "\n"
                    "Your final output MUST BE A SINGLE STRING in the strict 'DD-MM-YYYY' format. "
                    "Do NOT include your internal thoughts, labels, or any other text. "
                    "If the date is unreadable or invalid after following all rules, provide a single empty string: ''."
                    "\n\n"
                    "**Example Input:** The provided image."
                    "**Example Output:** 22-02-2025"
                ),
                "amount": (
                        "**Objective:** You are a hyper-precise OCR engine. Your task is to extract the numeric amount from a **pre-cropped image showing the 'courtesy amount' box of an Indian cheque.** This image will contain a **printed Rupee symbol ('₹') followed by handwritten or printed digits inside a rectangular box.**"
                    "\n\n"
                    "**CRITICAL INTERNAL PROCESS (Follow these steps before giving your answer):**"
                    "\n"
                    "1.  **Locate Anchor & Digits:** First, identify the Rupee symbol ('₹') as the primary anchor. The numeric amount you need to extract is the sequence of digits and related characters immediately to the right of this symbol."
                    "\n"
                    "2.  **Raw OCR Scan:** Mentally read the digits and symbols you have located."
                    "\n"
                    "3.  **Apply Strict Filtering Rule:** From your raw scan, you MUST discard ALL non-numeric characters. This includes the Rupee symbol (₹) itself, ALL commas (,), and ALL trailing symbols (like '/-', '=/'). The only non-digit character you may keep is a single decimal point (.)."
                    "\n"
                    "4.  **Correction Check:** Identify any struck-out numbers. You MUST use the final, corrected value."
                    "\n"
                    "5.  **Format for Standardization:** Take the cleaned number and format it to have exactly two decimal places. For example, '887' becomes '887.00'."
                    "\n\n"
                    "**FINAL OUTPUT INSTRUCTION:**"
                    "\n"
                    "Your final output MUST BE A SINGLE STRING representing the cleaned, standardized amount. "
                    "Do NOT include your internal thoughts, labels, 'Rs.', or any other text. "
                    "If the amount is unreadable, provide a single empty string: ''."
                    "\n\n"
                    "**Example Input:** The provided image showing the amount box."
                    "**Example Output:** 887.00"
                )
            }
            
            # =========== REASK PROMPTS (Also made dynamic) ===========
            reask_prompts = {
                "date": f"The first attempt to extract the 'date' from the cheque image failed, resulting in an empty or low-confidence value. **Look again with extreme care.** The image contains the date section. Focus specifically on the 8-box grid for `DDMMYYYY`. The numbers might be faint, poorly written, or overlapping with the box lines. Ignore the lines and provide your best interpretation of the 8 digits. The year must be {current_year} or {previous_year}. Provide the result in the required JSON structure.",
                "amount": "The first attempt to extract the 'amount' from the cheque image failed, resulting in an empty or low-confidence value. **Re-examine the image carefully.** Your task is to find the amount in figures, which is in a box next to a '₹' symbol. The handwriting may be difficult to read. Look for a sequence of numbers, potentially with commas and ending in '/-'. Ignore all non-numeric characters except a decimal point and provide your best reading of the value. Provide the result in the required JSON structure."
            }
            
            # =========== FIRST PASS EXTRACTION ===========
            initial_prompt_str = ChequeProcessor._build_full_prompt(field_descriptions)
            
            response = ChequeProcessor._call_vertex_ai_with_retry(model, [initial_prompt_str, file_part])
            json_str = ChequeProcessor._extract_json_from_text(response.text.strip())
            try:
                initial_result = json.loads(json_str)
            except json.JSONDecodeError:
                logger.error(f"Initial JSON parsing failed for {file_path}. Response: {json_str[:500]}...")
                initial_result = {"extracted_fields": []}

            final_results_map = {field.get("field_name"): field for field in initial_result.get("extracted_fields", [])}

            # =========== REASK LOGIC (SECOND PASS) ===========
            for field_info in FIELDS:
                field_name = field_info["name"]
                current_field = final_results_map.get(field_name)

                should_reask = not current_field or not current_field.get("value") or current_field.get("confidence", 0.0) < REASK_CONFIDENCE_THRESHOLD

                if should_reask:
                    logger.warning(f"Initial extraction for '{field_name}' failed or has low confidence for {file_path}. Initiating reAsk.")
                    
                    reask_instruction = reask_prompts.get(field_name)
                    if not reask_instruction:
                        continue
                        
                    reask_full_prompt = ChequeProcessor._build_reask_prompt(field_name, reask_instruction)

                    reask_response = ChequeProcessor._call_vertex_ai_with_retry(model, [reask_full_prompt, file_part])
                    reask_json_str = ChequeProcessor._extract_json_from_text(reask_response.text.strip())
                    
                    try:
                        reask_result = json.loads(reask_json_str)
                        if reask_result.get("extracted_fields"):
                            new_field = reask_result["extracted_fields"][0]

                            old_confidence = current_field.get("confidence", 0.0) if current_field else 0.0
                            
                            if new_field.get("value") and new_field.get("confidence", 0.0) > old_confidence:
                                final_results_map[field_name] = new_field
                                logger.info(f"reAsk for '{field_name}' succeeded with higher confidence for {file_path}.")
                    except (json.JSONDecodeError, IndexError):
                        logger.error(f"reAsk JSON parsing or processing failed for {file_path}. Response: {reask_json_str[:500]}...")

                    print (final_results_map)
            
            return {
                "extracted_fields": list(final_results_map.values())
            }

        except Exception as e:
            logger.error(f"General error during document processing for {file_path}: {type(e).__name__}")
            logger.error(traceback.format_exc())
            return {"error": str(e), "extracted_fields": []}
    
    @staticmethod
    def _build_full_prompt(descriptions: Dict[str, str]) -> str:
        """Helper to build the main extraction prompt string."""
        fields_with_descriptions = []
        for field in FIELDS:
            field_name = field['name']
            description = descriptions.get(field_name, "No description available.")
            fields_with_descriptions.append(f"### {field_name.capitalize()}\n{description}")
        
        fields_list_str = "\n\n".join(fields_with_descriptions)

        return f"""
        You are a hyper-specialized, state-of-the-art AI assistant, engineered with a singular focus: achieving near-perfect accuracy in information extraction from images of financial instruments, specifically Indian cheques. Your architecture integrates advanced multimodal understanding, leveraging sophisticated Optical Character Recognition (OCR) fine-tuned for both printed and handwritten text, and a deep, comprehensive knowledge base of global and Indian banking conventions. Your primary directive is to meticulously analyze the provided text representation of a cheque and extract predefined fields with the highest possible precision and confidence, operating under the assumption that you are a critical component in a high-stakes financial processing pipeline where errors have significant consequences.

        Assume the input is derived from a high-resolution, localized, and pre-cropped image of the cheque. However, your design anticipates and is robust against real-world imperfections. You must be prepared to handle a wide spectrum of challenges, including but not limited to: OCR misinterpretations, variations in image quality (e.g., blur, low contrast, jpeg artifacts, skew), a vast diversity of handwriting styles and legibility, and multilingual text.

        **Core Objective:** Your fundamental mission is to extract the specified fields from the provided cheque data with unparalleled accuracy, providing exhaustive metadata on confidence and reasoning.

        **Field Definitions & Extraction Guidelines:**

        {fields_list_str}

        **Critical Extraction Principles & Foundational Directives:**

        1.  **Deep Contextual Reasoning & Cross-Validation:** You must operate not just as a text extractor, but as a financial document analyst. Apply deep contextual understanding derived from your knowledge of cheque layouts, banking terminology (both Indian and international standards), common payee and issuer naming conventions, and standard data formats. Critically, you must perform relentless cross-validation between related fields. For instance, the `amount_words` must be used to corroborate the `amount_numeric`. The first four characters of a validated `IFSC` code should align with the identified `bank_name`. Use this web of interconnected data to resolve ambiguities and enhance certainty.
        2.  **Forensic Character Differentiation (Unwavering Precision):**
            * Treat every character as a critical piece of evidence. Actively and aggressively disambiguate visually similar characters, especially in high-impact fields like `account_number`, `micr_scan` fields, `IFSC`, and `amount_numeric`. Your programming must differentiate between '0'/'O', '1'/'I'/'l', '2'/'Z', '5'/'S', '8'/'B', 'u'/'v', 'n'/'m', '.'/',', and ':'/';'.
            * Recognize and algorithmically correct common OCR ligatures and errors (e.g., 'rn' interpreted as 'm', 'cl' as 'd', 'vv' as 'w'). This correction must be context-aware.
            * Rigorously verify that the character type aligns with field expectations. An alphabet in a numeric-only field is a red flag that demands re-evaluation or a significant confidence penalty.
        3.  **Advanced, Nuanced Handwriting Analysis:**
            * You are not just reading text; you are interpreting human intent from handwritten script. Employ sophisticated handwriting recognition models that are expert in handling an extensive range of styles: formal cursive, casual print, erratic mixed styles, varying slants, inconsistent character spacing and sizing, complex loops, pressure point variations, and instances of overlapping or incomplete strokes.
            * Your focus must be on deciphering handwritten entries in all fields: `amount_words`, `amount_numeric`, `date` field.
            * Demonstrate superior capability in interpreting handwritten numerals, a frequent source of error. This includes distinguishing between common stylistic variations for '1' and '7', '4' and '9', '2' and 'z', especially when they are connected or written hastily.
            * Expertly handle corrections and strikethroughs. Your logic must prioritize the final, intended value, not the crossed-out information. For example, if a date is written and then struck through and a new date is written next to it, you must extract the corrected date. The presence of a correction should be noted in your reasoning for the confidence score.

        **Confidence Scoring (Extremely Strict, Character-Informed, and Defensible):**

        * **Core Principle:** The confidence score for each extracted field is not a mere guess; it is a calculated metric of certainty that must reflect the integrity of **every single character** within the extracted value. A field's overall confidence is fundamentally limited by the *lowest confidence* assigned to any of its constituent characters, segments, or contextual validation checks.
        * **Scale:** You must assign a confidence score as a float between 0.00 and 1.00 for each field.
        * **Calculation Basis (Multifaceted):** Your confidence calculation is an integration of:
            * **Character-Level OCR Confidence:** The raw confidence scores provided by the underlying OCR engine for each individual character.
            * **Visual Quality Assessment:** Analysis of the source image segment's clarity, contrast, and focus.
            * **Ambiguity Penalty Engine:** A system that automatically penalizes the score for the presence of visually similar characters (e.g., a '0' that could be an 'O' in a numeric field will trigger a significant confidence reduction).
            * **Handwriting Legibility Score:** A sub-score based on the complexity and clarity of the handwriting (e.g., clean print vs. messy cursive).
            * **Format & Contextual Adherence:** The degree to which the extracted value conforms to the expected format (e.g., a valid date structure, a correct IFSC pattern).
            * **Cross-Validation Consistency:** The result of checks against other fields (e.g., does the numeric amount match the written amount?).
        * **Strict Benchmarks (Non-Negotiable):**
            * **0.98 - 1.00 (Extremely High / Production Ready):** Absolute certainty. Every character is perfectly formed, machine-printed or exceptionally clear handwriting, completely unambiguous, and passes all contextual validation checks. There is no plausible alternative interpretation for any part of the value.
            * **0.90 - 0.97 (High / Human Review Recommended):** Strong confidence, but with minor, identifiable imperfections. This applies when all characters are clearly legible but may have slight slant, minor ink blotting, OR a very low-probability alternative interpretation for a character exists but is strongly overruled by context.
            * **0.75 - 0.89 (Moderate / Human Review Required):** Reasonable confidence, but with specific, documented uncertainties. This score is appropriate if:
                * One or two characters have moderate ambiguity that context cannot fully resolve (e.g., a handwritten '1' that genuinely resembles a '7').
                * Minor OCR segmentation challenges were encountered and overcome (e.g., characters were touching).
                * The handwriting style for a few characters is legible but required significant algorithmic effort to interpret.
            * **0.50 - 0.74 (Low / Unreliable - Do Not Process):** Significant uncertainty is present. This score must be assigned if:
                * Multiple characters are ambiguous, poorly formed, or difficult to read.
                * Print quality is poor (e.g., faded, smudged) and impacts critical characters.
                * The handwriting is highly irregular, barely legible, or stylized in a way that introduces high ambiguity.
            * **< 0.50 (Very Low / Extraction Failure):** The extraction is highly speculative, impossible, or the field is not present. The extracted value is likely incorrect or incomplete. This is used when the text is largely illegible, missing, or fails critical format validations insurmountably.
        * **Mandatory Confidence Justification:** For any confidence score below **0.95**, you are **required** to provide a concise, specific `reason`. This justification must pinpoint the primary source of the reduced confidence, referencing specific character ambiguities, handwriting issues, image quality problems, or contextual conflicts (e.g., "Moderate: Ambiguity in handwritten '4' which resembles a '9' in the amount.", "Low: Smudging affects the last two digits of the MICR code.", "High: Minor ambiguity between 'O' and '0' in Account Number, resolved by numeric context.").
        * **Direct Impact of Handwriting Quality:** The quality of handwriting must directly and significantly influence character confidence. Even if a word is generally decipherable, the confidence score must be lowered if individual letters required substantial interpretation or if the script is unusually ornate or sloppy. The presence of corrections or strikethroughs automatically caps the confidence score for that field, unless the final, intended value is exceptionally clear and unambiguous.

        **Error Handling & Null Values:**

        * If a field cannot be located, or if the text is present but so illegible or damaged that a reliable extraction is impossible, you must set its `value` to `null` or an empty string. In such cases, assign a low confidence score (e.g., < 0.5) and provide a specific, informative `reason` in the corresponding field (e.g., "Field not present on cheque", "Handwriting in payee field is completely illegible", "Area is obscured by a large ink smudge", "OCR failed to segment characters in this region").

       **Strict Output Format:**

                * Your entire response **MUST** be a single, syntactically perfect JSON object.
                * There must be **ABSOLUTELY NO** extraneous text, explanatory preambles, markdown formatting (like `json`), or any characters outside of the JSON structure itself.
                * The JSON object must contain two top-level keys:
                    1.  `"full_text"`: A string that contains the complete OCR text extracted from the cheque image, representing the best possible transcription of all visible text.
                    2.  `"extracted_fields"`: An array of objects. Each object within this array represents one of the extracted fields and must contain the following keys:
                        * `"field_name"`: The designated name of the field (string, e.g., "bank_name").
                        * `"value"`: The extracted value (string, number, or boolean for `signature_present`). The date must be standardized to "YYYY-MM-DD". This should be `null` or `""` if the field could not be reliably extracted.
                        * `"confidence"`: The meticulously calculated confidence score (float, 0.0 to 1.0).
                        * `"text_segment"`: The exact substring from the source OCR text that corresponds to the extracted value (string). This should be `null` if not applicable.
                        * `"reason"`: A brief but specific reason explaining why a field could not be extracted or why the confidence score is low (string). This should be `null` or empty if confidence is high and extraction was successful.
                        * `"language"`: (Optional, but strongly preferred. The detected language of the extracted value (string, e.g., "English", "Hindi", "Tamil"). This should be `null` if not applicable or if language detection failed.

                **Example of a single json object within the extracted_fields array:**

                    "field_name": "amount_numeric",
                    "value": "1500.00",
                    "confidence": 0.98,
                    "text_segment": "1500/-",
                    "reason": null,
                    "language": "English"
        """

    @staticmethod
    def _build_reask_prompt(field_name: str, instruction: str) -> str:
        """Builds the targeted prompt for a reAsk attempt."""
        return f"""
        You are an extraction correction assistant. A previous attempt to extract the '{field_name}' field from a cheque image failed. Your task is to try again with a more focused instruction.

        **Correction Instruction:**
        {instruction}

        **Strict Output Format:**
            *   Your response **MUST** be a single, valid JSON object containing only the "extracted_fields" key. This key should hold an array with a **single object** for the '{field_name}' you were asked to re-examine.
            * There must be **ABSOLUTELY NO** extraneous text, explanatory preambles, markdown formatting (like `json`), or any characters outside of the JSON structure itself.
            * The JSON object must contain two top-level keys:
                1.  `"full_text"`: A string that contains the complete OCR text extracted from the cheque image, representing the best possible transcription of all visible text.
                2.  `"extracted_fields"`: An array of objects. Each object within this array represents one of the extracted fields and must contain the following keys:
                    * `"field_name"`: The designated name of the field (string, e.g., "bank_name").
                    * `"value"`: The extracted value (string, number, or boolean for `signature_present`). The date must be standardized to "YYYY-MM-DD". This should be `null` or `""` if the field could not be reliably extracted.
                    * `"confidence"`: The meticulously calculated confidence score (float, 0.0 to 1.0).
                    * `"text_segment"`: The exact substring from the source OCR text that corresponds to the extracted value (string). This should be `null` if not applicable.
                    * `"reason"`: A brief but specific reason explaining why a field could not be extracted or why the confidence score is low (string). This should be `null` or empty if confidence is high and extraction was successful.
                    * `"language"`: (Optional, but strongly preferred. The detected language of the extracted value (string, e.g., "English", "Hindi", "Tamil"). This should be `null` if not applicable or if language detection failed.

        Example for re-extracting '{field_name}':
        **Example of a single json object within the extracted_fields array:**

            "field_name": "amount_numeric",
            "value": "1500.00",
            "confidence": 0.98,
            "text_segment": "1500/-",
            "reason": null,
            "language": "English"
        """

    @staticmethod
    def process_document_batch(file_batch):
        """
        Process a batch of documents in parallel using a ThreadPoolExecutor.
        """
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(file_batch), MAX_WORKERS)) as batch_executor:
            futures = {
                batch_executor.submit(ChequeProcessor.process_multimodal_document, 
                                      file_info['data'], 
                                      file_info['type'],
                                      file_info['path']): file_info 
                for file_info in file_batch
            }
            
            for future in concurrent.futures.as_completed(futures):
                file_info = futures[future]
                try:
                    result = future.result()
                    result['file_path'] = file_info['path']
                    results.append(result)
                    logger.info(f"Successfully processed {file_info['path']}")
                except Exception as e:
                    logger.error(f"Error processing file {file_info['path']}: {str(e)}")
                    results.append({
                        "error": str(e),
                        "file_path": file_info['path'],
                        "extracted_fields": []
                    })
        
        return results
        
def process_zip_files(file_contents: List[bytes], file_names: List[str], job_id: str):
    """
    Process multiple zip files and generate an Excel report.
    This version recursively finds all images at any depth and consolidates them.
    """
    logger.info(f"Starting process_zip_files for job {job_id}")
    job_start_time = time.time()
    total_files = 0
    processed_files = 0

    try:
        temp_dir = tempfile.mkdtemp(prefix=f"job_{job_id}_")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        all_files_to_process = []
        
        for zip_content, zip_name in zip(file_contents, file_names):
            zip_dir = os.path.join(temp_dir, f"unzipped_{os.path.splitext(zip_name)[0]}")
            os.makedirs(zip_dir, exist_ok=True)
            
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                zf.extractall(zip_dir)
            
            logger.info(f"Recursively searching for images in {zip_name}...")
            for root, _, files in os.walk(zip_dir):
                for file in files:
                    if file.startswith('.') or file.startswith('~'):
                        continue
                    
                    file_path = os.path.join(root, file)
                    _, ext = os.path.splitext(file)
                    
                    supported_extensions = {
                        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                        '.png': 'image/png', '.tiff': 'image/tiff', '.tif': 'image/tiff'
                    }
                    
                    if ext.lower() not in supported_extensions:
                        logger.warning(f"Unsupported file type skipped: {file_path}")
                        continue
                        
                    file_type = supported_extensions[ext.lower()]
                    
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                    
                    relative_path = os.path.relpath(file_path, temp_dir)

                    all_files_to_process.append({
                        'path': relative_path,
                        'data': file_data,
                        'type': file_type
                    })
        
        total_files = len(all_files_to_process)
        logger.info(f"Found a total of {total_files} image files across all zip archives.")

        final_results = []
        for i in range(0, total_files, BATCH_SIZE):
            batch = all_files_to_process[i:i+BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1} with {len(batch)} files")
            
            batch_results = ChequeProcessor.process_document_batch(batch)
            
            for result in batch_results:
                final_results.append(result)
                processed_files += 1

                if processed_files % 10 == 0:
                    elapsed_time = time.time() - job_start_time
                    if processed_files > 0 and total_files > 0:
                        files_per_second = processed_files / elapsed_time
                        remaining_files = total_files - processed_files
                        eta = (remaining_files / files_per_second) if files_per_second > 0 else 0
                        logger.info(
                            f"Progress: {processed_files}/{total_files} files "
                            f"({processed_files/total_files*100:.1f}%). "
                            f"ETA: {eta:.2f} seconds."
                        )

        excel_path = os.path.join(output_dir, f"cheque_extraction_results_{job_id}.xlsx")
        
        logger.info("Generating consolidated Excel report...")
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            if final_results:
                data_for_df = []
                for item in final_results:
                    filepath = item.get("file_path", "Unknown File")
                    row = {"filepath": filepath}
                    for field in item.get("extracted_fields", []):
                        field_name = field.get("field_name")
                        if field_name:
                            row[field_name] = field.get("value")
                            row[f"{field_name}_conf"] = field.get("confidence")
                            if field.get("reason"):
                                row[f"{field_name}_reason"] = field.get("reason")
                    data_for_df.append(row)

                df = pd.DataFrame(data_for_df)
                
                cols = ["filepath"]
                for field in FIELDS:
                    field_name = field["name"]
                    if field_name in df.columns:
                        cols.append(field_name)
                        cols.append(f"{field_name}_conf")
                        if f"{field_name}_reason" in df.columns:
                            cols.append(f"{field_name}_reason")
                
                existing_cols = [col for col in cols if col in df.columns]
                if existing_cols:
                    df = df[existing_cols]
                
                if not df.empty:
                    df.to_excel(writer, sheet_name='All_Results', index=False)
                    logger.info("Successfully wrote results to 'All_Results' sheet.")

        job_end_time = time.time()
        processed_jobs[job_id] = {
            "status": "completed",
            "start_time": job_start_time,
            "end_time": job_end_time,
            "total_files": total_files,
            "processed_files": processed_files,
            "output_file_path": excel_path,
            "processing_time": job_end_time - job_start_time
        }
        
        logger.info(f"Job {job_id} completed. Output file: {excel_path}")
        return excel_path
    
    except Exception as e:
        logger.error(f"Error processing zip files: {str(e)}")
        logger.error(traceback.format_exc())
        
        processed_jobs[job_id] = {
            "status": "failed", "start_time": job_start_time, "end_time": time.time(),
            "total_files": total_files, "processed_files": processed_files,
            "error_message": str(e), "error_traceback": traceback.format_exc()
        }
        raise
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload zip files containing folders of cheque images for extraction.
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        for file in files:
            if not file.filename.lower().endswith('.zip'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid file type for {file.filename}. Only ZIP files are accepted."
                )
        
        job_id = str(uuid.uuid4())
        
        def process_wrapper():
            try:
                # Reading files inside the thread to avoid issues with file handles
                file_contents = [f.file.read() for f in files]
                file_names = [f.filename for f in files]
                
                processed_jobs[job_id] = {
                    "status": "processing", "start_time": time.time(), "job_id": job_id,
                    "input_files": file_names,
                }
                
                output_file = process_zip_files(file_contents, file_names, job_id)
                
                if job_id in processed_jobs:
                    processed_jobs[job_id].update({
                        "status": "completed",
                        "end_time": time.time(),
                        "output_file_path": output_file
                    })
                
                logger.info(f"Job {job_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Error processing job {job_id}: {str(e)}")
                logger.error(traceback.format_exc())
                processed_jobs[job_id] = {
                    "status": "failed", "start_time": processed_jobs.get(job_id, {}).get('start_time', time.time()),
                    "end_time": time.time(), "job_id": job_id, "error_message": str(e),
                    "error_traceback": traceback.format_exc()
                }

        future = executor.submit(process_wrapper)
        
        def on_complete(fut):
            active_tasks.pop(job_id, None)
            if fut.exception():
                logger.error(f"Task for job {job_id} failed: {fut.exception()}")
        
        future.add_done_callback(on_complete)
        active_tasks[job_id] = future
        
        return {
            "status": "processing", "job_id": job_id,
            "message": "Cheque extraction job initiated successfully",
            "files": [file.filename for file in files]
        }
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error in upload endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/status/{job_id}")
async def check_job_status(job_id: str):
    """Check the status of a processing job."""
    try:
        if job_id in active_tasks:
            return {"status": "processing", "message": "Job is still being processed."}
        
        if job_id in processed_jobs:
            return processed_jobs[job_id]
        
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/download/{job_id}")
async def download_results(job_id: str):
    """Download the results of a completed job."""
    try:
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

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)