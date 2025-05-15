from app.constants import FIELDS

# Field descriptions for prompt generation
# (Copied verbatim from the original, ensure these are optimal for your model)
FIELD_DESCRIPTIONS = {
    "bank_name": (
        "**Objective:** Accurately extract the official name of the issuing bank.\n"
        # ... (rest of bank_name description)
         "**Output:** The standardized, official bank name."
    ),
    "bank_branch": (
        "**Objective:** Extract the specific branch name or location identifier.\n"
        # ... (rest of bank_branch description)
        "**Output:** The full branch name/location string as it appears, combined across lines if necessary."
    ),
    "account_number": (
        "**Objective:** Extract the customer's bank account number.\n"
        # ... (rest of account_number description)
        "**Output:** The extracted account number sequence (digits/alphanumerics only), with separators removed. If definitively not present, output 'Not Found'."
    ),
    "date": (
        "**Objective:** Extract the issue date and standardize it.\n"
        # ... (rest of date description)
        "**Output Format:** **Strictly YYYY-MM-DD.** Convert all valid inputs to this format."
    ),
    "payee_name": (
        "Objective: Extract the complete name and any associated payment instructions for the recipient (person or entity) to whom the cheque is payable.\n"
        # ... (rest of payee_name description)
        "Output: The complete text extracted from the payee line(s) as a single string. This string should include the recipient's name along with any directly associated and embedded bank account numbers, bank names, or other payment instructions if they are present as part of the continuous text on the payee line(s). If the information spans multiple lines, these should be concatenated, typically separated by a single space."
    ),
    "amount_words": (
        "**Objective:** Extract the cheque amount written in words (legal amount).\n"
        # ... (rest of amount_words description)
        "**Output:** The full amount in words string."
    ),
    "amount_numeric": (
        "**Objective:** Extract the cheque amount written in figures (courtesy amount).\n"
        # ... (rest of amount_numeric description)
        "**Output:** The cleaned, purely numeric amount string (e.g., '1500.00', '1500.50', '12000')."
    ),
    "issuer_name": (
        "Objective: Extract the name(s) of the account holder(s) or the company name issuing the cheque (payer).\n"
        # ... (rest of issuer_name description)
        "Output: The extracted issuer name(s) or company name. If only a signature exists and no identifiable printed or stamped name is found in the designated areas, output 'Not Found'."
    ),
    "micr_scan_instrument_number": (
        "**Objective:** Extract the 6-digit cheque serial number (instrument number) from the E-13B MICR line with utmost precision.\n"
        # ... (rest of micr_scan_instrument_number description)
        "**Output:** A string containing exactly 6 numeric digits. Null if criteria not met."
    ),
    "micr_scan_payee_details": (
        "**Objective:** Extract the 9-digit bank sort code (City-Bank-Branch identifier, often referred to as the MICR code itself) from the E-13B MICR line. Note: Per user instruction, this field is named 'micr_scan_payee_details', but it represents the bank's routing information on Indian cheques.\n"
        # ... (rest of micr_scan_payee_details description)
        "**Output:** A string containing exactly 9 numeric digits. Null if criteria not met."
    ),
    "micr_scan_micr_acno": (
        "**Objective:** Extract a 6-digit account-related or secondary transaction code from the E-13B MICR line, if structurally present.\n"
        # ... (rest of micr_scan_micr_acno description)
        "**Output:** A string containing exactly 6 numeric digits if present and correctly parsed, or '000000' if structurally absent as per rules. Null for actual read errors of a present field that cannot be confidently extracted."
    ),
    "micr_scan_instrument_type": (
        "**Objective:** Extract the 2-digit transaction code or instrument type from the very end of the E-13B MICR line.\n"
        # ... (rest of micr_scan_instrument_type description)
        "**Output:** A string containing exactly 2 numeric digits. Null if criteria not met."
    ),
    "signature_present": (
        "**Objective:** Determine if a handwritten signature exists in the designated area.\n"
        # ... (rest of signature_present description)
        "**Output:** **Strictly 'YES' or 'NO'.**"
    ),
    "IFSC": (
        "**Objective:** Extract the 11-character Indian Financial System Code.\n"
        # ... (rest of IFSC description)
        "**Output:** The validated 11-character IFSC code. If validation fails, output 'Error' / 'Not Found'."
    ),
    "currency": (
        "**Objective:** Identify the currency of the transaction.\n"
        # ... (rest of currency description)
        "**Output:** The standard 3-letter ISO 4217 currency code (e.g., 'INR', 'USD', 'EUR')."
    )
}
# Ensure all descriptions are filled in as they were in your original code.
# For brevity, I've truncated them here.

SIGNATURE_EXTRACTION_PROMPT = """
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

def get_cheque_extraction_prompt():
    doc_fields_names = [field['name'] for field in FIELDS]
    fields_with_descriptions = []
    for field_name in doc_fields_names:
        description = FIELD_DESCRIPTIONS.get(field_name, f"Description for {field_name} not found.")
        fields_with_descriptions.append(f"- {field_name}: {description}")
    
    fields_list_str = "\n".join(fields_with_descriptions)

    return f"""You are an expert AI assistant specializing in high-accuracy information extraction from scanned cheque images, leveraging advanced multimodal understanding, sophisticated OCR, and deep domain knowledge of global banking practices, particularly Indian cheques. Your task is to meticulously analyze the provided text representation of a cheque and extract specific fields with maximum precision and confidence.

Assume the input text originates from a high-resolution scan, but be prepared to handle potential OCR errors, variations in image quality (blur, low contrast, skew), handwriting ambiguities, and multilingual content.

**Core Objective:** Extract the specified fields from the cheque data.

**Field Definitions & Extraction Guidelines:**

{fields_list_str}

**Critical Extraction Principles & Guidelines:**

1.  **Contextual Reasoning:** Apply deep contextual understanding. Use knowledge of cheque layouts, banking terminology (Indian and international), common payee names, and standard formats to interpret information correctly. Cross-validate information between fields (e.g., amount words vs. numeric amount, bank name vs. IFSC/MICR).
2.  **Character Differentiation (Precision Focus):**
    * Actively disambiguate visually similar characters (0/O, 1/I/l, 2/Z, 5/S, 8/B, ./,, :/; etc.). Pay extreme attention in critical fields like Account Numbers, MICR, IFSC, and Amounts.
    * Recognize common OCR ligatures/errors (rn/m, cl/d, vv/w) and correct them based on context.
    * Verify character types against field expectations (e.g., digits in `account_number`, `amount_numeric`, `micr_code`, `IFSC`; predominantly letters in names).
3.  **Advanced Handwriting Analysis:**
    * Employ sophisticated handwriting recognition models capable of handling diverse styles (cursive, print, mixed), varying slant, inconsistent spacing/size, loops, pressure points, and potential overlaps or incompleteness.
    * Specifically address challenges in handwritten: `payee_name`, `amount_words`, `amount_numeric`, `date`, `issuer_name`, and `signature_present` assessment.
    * Accurately interpret handwritten numbers, distinguishing styles for '1'/'7', '4'/'9', '2', etc., even when connected.
    * Handle corrections (strikethroughs): Prioritize the final, intended value, not the crossed-out text. If a date is corrected, extract the corrected date.
4.  **Multilingual & Mixed-Script Processing:**
    * Accurately identify and transcribe text in multiple languages, primarily English and major Indian languages (Hindi, Kannada, Telugu, Tamil, Punjabi, Bengali, etc.).
    * Specify the detected language for fields prone to multilingual content (`payee_name`, `amount_words`, `issuer_name`) if not English.
    * Apply script-specific character differentiation rules (e.g., Devanagari ण/ज़, த/த; Tamil ன/ண, ர/ற; similar forms in Telugu/Kannada/Bengali/Assamese).
    * Handle code-switching (mixing scripts/languages) within a single field value where appropriate.
    * Recognize and correctly transcribe Indian language numerals if present.
5.  **MICR Code Extraction:**
    * Target the E-13B font sequence at the cheque bottom.
    * Extract **digits only (0-9)**. Explicitly **exclude** any non-digit symbols or delimiters (like ⑆, ⑈, ⑇).
    * Validate the typical 9-digit structure for Indian cheques (CCCBBBAAA - City, Bank, Branch). Note variations if necessary.
    * Ensure high confidence differentiation of MICR's unique blocky characters.
6.  **Date Extraction & Standardization:**
    * Locate the date, typically top-right.
    * Recognize various formats (DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD, DD-Mon-YYYY, etc.) including handwritten variations.
    * Handle partial pre-fills (e.g., printed "20" followed by handwritten "24").
    * Accurately parse day, month, and year, resolving ambiguity using context (assume DD/MM for India unless clearly otherwise) and proximity to the likely processing date (cheques are typically valid for 3-6 months).
    * Standardize the final output strictly to **YYYY-MM-DD** format. If the date is invalid or ambiguous (e.g., Feb 30), flag it.
7.  **Amount Validation:** Ensure `amount_numeric` and `amount_words` correspond logically. Note discrepancies if unavoidable. Extract numeric amount precisely, including decimals if present.
8.  **Signature Detection:** Assess the presence of handwritten, free-flowing ink strokes in the typical signature area (bottom right, above MICR). Output only "YES" or "NO". Do not attempt to read the signature text itself for the `signature_present` field.

**Confidence Scoring (Strict, Character-Informed):**

* **Core Principle:** The overall confidence score for each field MUST reflect the system's certainty about **every single character** comprising the extracted value. The field's confidence is heavily influenced by the *lowest* confidence assigned to any of its critical constituent characters or segments during the OCR/interpretation process.
* **Scale:** Assign a confidence score (float, 0.00 to 1.00) for each extracted field.
* **Calculation Basis:** This score integrates:
    * OCR engine's internal character-level confidence values.
    * Visual clarity and quality of the source text segment.
    * Ambiguity checks (e.g., similar characters like 0/O, 1/I).
    * Handwriting legibility (individual strokes, connections).
    * Adherence to expected field format and context (e.g., a potential 'O' in a numeric field drastically lowers confidence).
    * Cross-validation results (e.g., amount words vs. numeric).
* **Strict Benchmarks:**
    * **0.98 - 1.00 (Very High):** Near certainty. All characters are perfectly clear, unambiguous, well-formed (print or handwriting), and fully context-compliant. No plausible alternative interpretation exists for any character.
    * **0.90 - 0.97 (High):** Strong confidence. All characters are clearly legible, but minor imperfections might exist (e.g., slight slant, minor ink variation) OR very low-probability alternative character interpretations exist but are strongly ruled out by context.
    * **0.75 - 0.89 (Moderate):** Reasonable confidence, but with specific, identifiable uncertainties. This applies if:
        * One or two characters have moderate ambiguity (e.g., a handwritten '1' that *could* be a '7', a slightly unclear 'S' vs '5').
        * Minor OCR segmentation issues were overcome (e.g., slightly touching characters).
        * Legible but challenging handwriting style for a character or two.
    * **0.50 - 0.74 (Low):** Significant uncertainty exists. This applies if:
        * Multiple characters are ambiguous or difficult to read.
        * Poor print quality (faded, smudged) affects key characters.
        * Highly irregular or barely legible handwriting is involved.
        * Strong conflicts exist (e.g., amount words clearly mismatch numeric, but an extraction is still attempted).
    * **< 0.50 (Very Low / Unreliable):** Extraction is highly speculative or impossible. The field value is likely incorrect or incomplete. Assign this if the text is largely illegible, completely missing, or fails critical format validations無法克服地 (insurmountably).
* **Confidence Justification:** **Mandatory** for any score below **0.95**. Briefly explain the *primary reason* for the reduced confidence, referencing specific character ambiguities, handwriting issues, print quality, or contextual conflicts (e.g., "Moderate: Handwritten '4' resembles '9'", "Low: MICR digits '8' and '0' partially smudged", "High: Minor ambiguity between 'O'/'0' in Acc No, resolved by numeric context").
* **Handwriting Impact:** Directly link handwriting quality to character confidence. Even if a word is *generally* readable, confidence drops if individual letters require significant interpretation effort. Corrections/strikethroughs automatically cap confidence unless the final value is exceptionally clear.

**Error Handling:**

* If a field cannot be found or reliably extracted, set its value to `null` or an empty string, assign a low confidence score (e.g., < 0.5), and provide a specific `reason` (e.g., "Field not present", "Illegible handwriting", "Smudged area", "OCR segmentation failed").

**Output Format:**

* Your response **MUST** be a single, valid JSON object.
* **Do NOT** include any explanatory text, markdown formatting, or anything outside the JSON structure.
* The JSON should have two top-level keys:
    1.  `"full_text"`: A string containing the entire OCR text extracted from the cheque, as accurately as possible.
    2.  `"extracted_fields"`: An array of objects. Each object represents an extracted field and must contain:
        * `"field_name"`: The name of the field (string, e.g., "bank_name").
        * `"value"`: The extracted value (string, number, or boolean for `signature_present`). Standardize date to "YYYY-MM-DD". Null or "" if not found/extractable.
        * `"confidence"`: The confidence score (float, 0.0-1.0).
        * `"text_segment"`: The exact text substring from the source OCR corresponding to the extracted value (string). Null if not applicable.
        * `"reason"`: A brief reason if the field could not be extracted or confidence is low (string). Null or empty otherwise.
        * `"language"`: (Optional, but preferred for `payee_name`, `amount_words`, `issuer_name`) The detected language of the extracted value (string, e.g., "English", "Hindi", "Tamil"). Null if not applicable or detection failed.

**Example extracted_fields object will contain all these fields with example values like:
    "field_name": "amount_numeric",
    "value": "1500.00",
    "confidence": 0.98,
    "text_segment": "1500/-",
    "reason": null,
    "language": "English"

IMPORTANT: Your response must be a valid JSON object and NOTHING ELSE. No explanations, no markdown code blocks.
"""