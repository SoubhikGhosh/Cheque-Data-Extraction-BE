from app.constants import FIELDS

# Field descriptions for prompt generation
# (Copied verbatim from the original, ensure these are optimal for your model)
FIELD_DESCRIPTIONS = {
    "bank_name": (
        "**Objective:** Accurately extract the official name of the issuing bank.\n"
        "**Primary Location Strategy:** Focus EXCLUSIVELY on the most prominent bank name text, typically located in the **top-left quadrant** of the cheque. This often corresponds to the bank's primary logo/branding.\n"
        "**Disambiguation Rules:**\n"
        "  1. If multiple bank names appear (e.g., clearing bank stamps), STRICTLY prioritize the top-left issuing bank name.\n"
        "  2. Actively differentiate the bank name from the `payee_name` based on typical location and keywords ('PAY TO').\n"
        "**Extraction Method:** Utilize visual layout analysis combined with knowledge of major Indian and international bank names. Cross-reference with the first 4 characters of the `IFSC` code if available and unambiguous.\n"
        "**Data Cleansing:** Normalize variations (e.g., 'HDFC Bank' -> 'HDFC BANK LTD'). Validate against a comprehensive list of known Indian bank names. Treat 'HDFC BANK LTD' as a correct and valid name.\n"
        "**Handling Poor Quality:** Employ image enhancement techniques (contrast, binarization) specifically on the target area if scan quality is low.\n"
        "**Output:** The standardized, official bank name."
    ),
    "bank_branch": (
        "**Objective:** Extract the specific branch name or location identifier.\n"
        "**Primary Location Strategy:** Search for text indicating a location (city, area, branch designation) directly **below or adjacent to the extracted `bank_name`**.\n"
        "**Content:** Expect branch names, city names, area codes, or combinations. Can span multiple lines.\n"
        "**Extraction Method:**\n"
        "  1. Look for explicit labels like 'Branch:', 'Br.', 'IFSC:' (branch info often follows IFSC).\n"
        "  2. Capture text immediately following the `bank_name` even without explicit labels.\n"
        "  3. **Crucially, combine text from multiple consecutive lines** if the branch address appears split.\n"
        "  4. Apply text segmentation to isolate branch info from surrounding elements (e.g., logos, address lines unrelated to branch).\n"
        "**Handling Partial Reads:** If only partial text is clear, use context (common Indian city/area names) to infer the most likely branch identifier. You can also use the IFSC Code to get the branch details in case of illegible text.\n"
        "**Output:** The full branch name/location string as it appears, combined across lines if necessary."
    ),
    "account_number": (
        "**Objective:** Extract the customer's bank account number.\n"
        "**Location Strategy:** Search in priority order:\n"
        "  1. Near explicit labels: 'A/C No.', 'Account No.', 'SB Acct', 'Acct No.', etc.\n"
        "  2. Often located near the `payee_name` line or below bank/branch details.\n"
        "  3. Check the footer/MICR area, especially on non-standard formats (Drafts, Manager's Cheques).\n"
        "  4. Scan both horizontal and potentially vertical orientations near the edges.\n"
        "**Format:** Primarily numeric (typically 9-18 digits). May contain hyphens or spaces (which **must be removed** in the final output). Can occasionally be alphanumeric for specific banks/account types.\n"
        "**Extraction Method:** Identify digit sequences matching expected lengths associated with account labels or typical locations. Apply robust digit/character recognition (0/O, 1/I, 5/S, 8/B differentiation).\n"
        "**Handling Special Formats:** Recognize that formats like PAYINST DRAFT or Manager's Cheques may lack explicit 'A/C No.' labels; rely on typical location for these formats.\n"
        "**Output:** The extracted account number sequence (digits/alphanumerics only), with separators removed. If definitively not present, output 'Not Found'."
    ),
    "date": (
        "**Objective:** Extract the issue date and standardize it.\n"
        "**Primary Location Strategy:** Target the **top-right corner**, typically within designated DD MM YYYY boxes.\n"
        "**Input Format Handling:** Recognize and parse various common formats: DDMMYYYY, DD/MM/YYYY, DD-MM-YY, DD.MM.YYYY, DD Mon YYYY, handwritten variations, dates printed over boxes, and partial pre-fills (e.g., printed '20__' with handwritten '24').\n"
        "**Extraction Method:**\n"
        "  1. Segment the Day (DD), Month (MM), and Year (YYYY/YY) components.\n"
        "  2. Apply specific OCR techniques for both printed and handwritten digits within the date area.\n"
        "  3. Use image processing to handle characters overlapping box lines.\n"
        "  4. If multiple dates are present (e.g., stamp vs handwritten), prioritize the main handwritten/typed date in the designated boxes.\n"
        "  5. Validate the extracted date as a plausible calendar date (e.g., day <= 31, month <= 12) within a reasonable past window (e.g., last 12 months unless context suggests older validity).\n"
        "**Output Format:** **Strictly YYYY-MM-DD.** Convert all valid inputs to this format."
    ),
    "payee_name": (
        "Objective: Extract the complete name and any associated payment instructions for the recipient (person or entity) to whom the cheque is payable.\n"
        "Primary Location Strategy: Target the text immediately following keywords such as 'PAY', 'Pay To', 'Pay to the order of', or 'Payee:'. This information is typically found on one or more lines situated below the bank's details and above the amount_words section.\n"
        "Content: The payee information is most often handwritten but can occasionally be typed or stamped. It primarily consists of an individual's name or a company/organization name. This section may include:\n"
        "  * Titles (e.g., Mr., Ms., Mrs., Dr., M/s).\n"
        "  * Company suffixes (e.g., Pvt Ltd, Ltd, Inc., LLP).\n"
        "  * Crucially, the payee line(s) can also embed specific payment instructions directly within or alongside the name. This includes details like the payee's bank account number, bank name, and sometimes even an IFSC code (e.g., 'JOHN PETER DOE A/C 1234567890 XYZ BANK', 'ACME CORP A/C 987654321 TO BE CREDITED TO PQR BANK IFSC ABCD0123456', 'M/S INFOTECH SOLUTIONS PAY YOURSELVES A/C NO XXXXX').\n"
        "  * The text can span a single line or wrap across multiple lines.\n"
        "Extraction Method:\n"
        "  1.  Comprehensive Text Block Capture: Identify and capture ALL text written on the designated payee line(s). This capture should begin immediately after the 'PAY' (or equivalent) keyword and extend for the full width of the line(s) dedicated to the payee, stopping before the amount_words or amount_numeric sections begin. Ensure to include any embedded account numbers, bank names, or other specific instructions if they form a continuous part of the text on these payee line(s). If the information spans multiple lines, it should be captured and concatenated logically (usually with a space).\n"
        "  2.  Advanced Handwriting Analysis (Emphasis on Cursive and Connected Scripts): Employ sophisticated handwriting recognition models. These models must be highly proficient in interpreting diverse handwriting styles, with particular strength in:\n"
        "      * Complex Cursive and Semi-Cursive Scripts: Accurately deciphering flowing, connected, and looped characters.\n"
        "      * Connected Lettering: Handling letters that are joined together, which is common in cursive.\n"
        "      * Variable Slant, Size, and Spacing: Adapting to inconsistencies in character formation.\n"
        "      * Ligatures and Common Ambiguities: Recognizing and correctly interpreting common ligatures (e.g., 'rn' vs 'm', 'cl' vs 'd') and handwriting ambiguities prevalent in both English and Indian language scripts.\n"
        "  3.  High-Precision Character Differentiation: Given that names and account details are highly sensitive to errors, the OCR process must apply maximum precision when differentiating visually similar characters. This is critical for both printed and handwritten text. Examples include (but are not limited to):\n"
        "      * Handwritten: 'u'/'v'/'w', 'n'/'m'/'h', 'a'/'o'/'u', 'e'/'c'/'o', 'l'/'t'/'f', 'i'/'j'/'l', 'r'/'s', 'g'/'y'/'q'.\n"
        "      * General: '0'/'O', '1'/'I'/'l', '2'/'Z', '5'/'S', '8'/'B'.\n"
        "      Contextual understanding (e.g., common naming patterns, keywords like 'A/c') should be used to aid disambiguation, but the foundation must be robust character-level recognition.\n"
        "  4.  Multilingual & Mixed-Script Processing: Accurately identify, transcribe, and specify the detected language for payee information, especially if it involves English or major Indian languages (e.g., Hindi, Tamil, Telugu). Handle instances of mixed-script content within the payee line where applicable.\n"
        "  5.  Structural Awareness: While capturing the full text, maintain awareness of potential structures like 'Name part' then 'A/c No.' then 'Bank Details'. This awareness can aid in the interpretation, even if the final output is a single string.\n"
        "  6.  Contextual Disambiguation: Clearly distinguish the payee_name block from other fields like the issuer_name based on its specific location and the preceding 'PAY' (or equivalent) keyword.\n"
        "Output: The complete text extracted from the payee line(s) as a single string. This string should include the recipient's name along with any directly associated and embedded bank account numbers, bank names, or other payment instructions if they are present as part of the continuous text on the payee line(s). If the information spans multiple lines, these should be concatenated, typically separated by a single space."
    ),
    "amount_words": (
        "**Objective:** Extract the cheque amount written in words (legal amount).\n"
        "**Primary Location Strategy:** Target the line(s) typically starting below the `payee_name`, often beginning with 'Rupees' or the currency name.\n"
        "**Content:** Primarily handwritten text representing the numeric value, potentially spanning two lines, often ending with 'Only'. Can include fractional units ('Paise') and Indian numbering terms ('Lakh', 'Crore').\n"
        "**Extraction Method:**\n"
        "  1. Capture the *entire* text phrase from the start (e.g., 'Rupees') to the end (e.g., 'Only').\n"
        "  2. Apply advanced handwriting recognition.\n"
        "  3. **Handle Multilingual Text:** Recognize and correctly interpret Indian language number words (e.g., 'हजार', 'लाख', 'കോടി') and currency names ('रुपये').\n"
        "  4. **Validate:** Use the recognized `amount_numeric` as a strong cross-validation signal to confirm the accuracy of the extracted words.\n"
        "  5. Handle hyphenation and line breaks correctly if the amount spans multiple lines.\n"
        "**Output:** The full amount in words string."
    ),
    "amount_numeric": (
        "**Objective:** Extract the cheque amount written in figures (courtesy amount).\n"
        "**Primary Location Strategy:** Target the designated box or area on the **right-middle side**, often preceded/followed by a currency symbol or code.\n"
        "**Format:** Numeric digits, possibly with commas (thousands) and a period (decimal). Often ends with '/-' or '.00'. Currency symbols (₹, $) may be adjacent.\n"
        "**Extraction Method:**\n"
        "  1. Isolate the numeric digits within the designated amount box/area.\n"
        "  2. Apply robust digit recognition, handling potential confusion (1/7, 4/9, etc.) and various handwriting styles.\n"
        "  3. **Crucially, PREPROCESS the extracted string:** Remove any currency symbols (₹, $, INR), thousands separators (,), and common trailing characters ('/-') BEFORE outputting.\n"
        "  4. Retain the decimal separator (.) and subsequent digits if present (e.g., '1500.50').\n"
        "  5. Standardize formats: '1500' and '1500.00' should both be represented consistently if required (e.g., always include '.00' or never include if zero). Clarify desired output format.\n"
        "  6. **Validate:** Use the recognized `amount_words` for cross-validation.\n"
        "**Output:** The cleaned, purely numeric amount string (e.g., '1500.00', '1500.50', '12000')."
    ),
    "issuer_name": (
        "Objective: Extract the name(s) of the account holder(s) or the company name issuing the cheque (payer).\n"
        "Primary Location Strategy: Search the area below the signature space, typically on the bottom-right, positioned above the MICR line. Also, check for printed company names, potentially located in the top-left quadrant under the bank details.\n"
        "Content: Look for printed text or rubber stamp impressions representing:\n"
        "  * An individual's name.\n"
        "  * Multiple individuals' names (for joint accounts), often separated by 'AND' or similar conjunctions.\n"
        "  * A company or organization name. This might be prefixed with terms like 'FOR' (e.g., 'FOR ABC ENTERPRISES').\n"
        "Extraction Method:\n"
        "  1. Focus OCR on identifying printed or stamped text in the primary location(s). Do NOT attempt to read the handwritten signature itself for this field.\n"
        "  2. Handle Signatures Overlap: If a signature overlaps the printed/stamped name, apply image segmentation techniques to isolate the underlying text/stamp from the signature strokes.\n"
        "  3. Identify Company Names: If the text follows the pattern 'FOR [Company Name]', extract '[Company Name]'.\n"
        "  4. Identify Joint Accounts: If multiple distinct names are clearly printed in the issuer area, capture all names (e.g., 'John Doe AND Jane Doe', 'Name1 / Name2'). Combine them as they appear.\n"
        "  5. Disambiguate: CRITICALLY differentiate the issuer from the payee_name based on location (Payee is after 'PAY TO', Issuer is near signature/bottom).\n"
        "Output: The extracted issuer name(s) or company name. If only a signature exists and no identifiable printed or stamped name is found in the designated areas, output 'Not Found'."
    ),
    "micr_scan_instrument_number": (
        "**Objective:** Extract the 6-digit cheque serial number (instrument number) from the E-13B MICR line with utmost precision.\n"
        "**Overall MICR Structure Expectation:** The full MICR line on an Indian cheque is generally understood to be composed of segments typically arranged in a specific order. This field, `micr_scan_instrument_number`, is the **first segment** in this sequence.\n"
        "**Positional Definition:** This field is **strictly the first distinct numeric group** identifiable as E-13B digits, located at the **absolute beginning** of the MICR encoded data strip at the bottom of an Indian cheque.\n"
        "**Pattern and Delimiters (Strict Interpretation):**\n"
        "  * Actively search for a sequence of **exactly 6 numeric E-13B digits (0-9)**.\n"
        "  * This sequence is **critically expected to be enclosed by MICR 'On-Us' symbols (⑈)** on both sides, forming the pattern: `⑈DDDDDD⑈` (where D is a digit).\n"
        "  * **Preceded by:** Nothing; this segment marks the start of the MICR line.\n"
        "  * **Followed by:** The `micr_scan_payee_details` segment. The end of the `micr_scan_instrument_number` segment is defined by its trailing `⑈` symbol. The `micr_scan_payee_details` segment is expected to begin immediately after this, typically commencing with its own leading `⑈` symbol.\n"
        "  * The digits themselves might have minor print spacing variations (e.g., '005 656' or '005656'), but the final extracted value **must be a contiguous 6-digit string** with all internal/external spaces removed.\n"
        "  * **Leading zeros are integral** and MUST be preserved (e.g., '005656' is correct, not '5656').\n"
        "**Extraction Method - CRITICAL STEPS & ROBUSTNESS:**\n"
        "  1.  **E-13B OCR Specialization:** Employ OCR highly tuned for E-13B font. Generic OCR is insufficient.\n"
        "  2.  **Locate Initial Segment:** Identify the segment at the very start of the MICR line. Prioritize a 6-digit sequence explicitly matching the `⑈DDDDDD⑈` pattern.\n"
        "  3.  **Aggressive Digit Filtering:** Extract **ONLY the 6 numeric E-13B digits**. Rigorously exclude the `⑈` symbols and any other non-digit characters, OCR noise, or artifacts from the final value.\n"
        "  4.  **Delimiter Integrity Check:** If one or both `⑈` symbols are missing or misrecognized by OCR, but a clear, isolated 6-digit E-13B numeric sequence is unambiguously present at the very start of the MICR line, the digits may be extracted, but confidence must be lowered with a justification (e.g., 'Missing leading/trailing ⑈ delimiter for instrument_number'). If ambiguity arises due to missing delimiters (e.g., unclear start of sequence), prioritize not extracting or assign very low confidence.\n"
        "  5.  **Print Quality Handling:** If E-13B digits are smudged or broken but still interpretable as specific digits with high probability, extract them and reduce confidence, noting the specific imperfection (e.g., 'Digit '0' in instrument_number partially smudged'). If a digit is entirely illegible or ambiguous between multiple possibilities, this sub-field extraction should fail or have extremely low confidence.\n"
        "**Error Handling:** If a clear 6-digit E-13B sequence, ideally matching the `⑈DDDDDD⑈` pattern, cannot be confidently identified at the start of the MICR line, or if it contains non-removable non-numeric characters, this field must be null, with confidence < 0.5 and reason 'Instrument number segment not found or illegible'.\n"
        "**Output:** A string containing exactly 6 numeric digits. Null if criteria not met."
    ),
    "micr_scan_payee_details": (
        "**Objective:** Extract the 9-digit bank sort code (City-Bank-Branch identifier, often referred to as the MICR code itself) from the E-13B MICR line. Note: Per user instruction, this field is named 'micr_scan_payee_details', but it represents the bank's routing information on Indian cheques.\n"
        "**Overall MICR Structure Expectation:** This field is the **second primary segment** in the MICR line, following the `micr_scan_instrument_number`.\n"
        "**Positional Definition:** This 9-digit numeric group **must immediately follow the complete `micr_scan_instrument_number` segment** (i.e., after the `micr_scan_instrument_number`'s trailing `⑈` symbol).\n"
        "**Pattern and Delimiters (Strict Interpretation):**\n"
        "  * Search for a sequence of **exactly 9 numeric E-13B digits (0-9)**.\n"
        "  * This sequence is **critically expected to be enclosed by a leading MICR 'On-Us' symbol (⑈) and a trailing MICR 'Transit' symbol (⑆)**, forming the pattern: `⑈DDDDDDDDD⑆`.\n"
        "  * **Preceded by:** The complete `micr_scan_instrument_number` segment (which ends with `⑈`). This `micr_scan_payee_details` segment starts with its own `⑈` delimiter.\n"
        "  * **Followed by:** The `micr_scan_micr_acno` segment (which is expected to start with `⑆`) or, if `micr_scan_micr_acno` is structurally absent, by the `micr_scan_instrument_type` segment. The end of the `micr_scan_payee_details` segment is defined by its trailing `⑆` symbol.\n"
        "  * Internal print spacing variations are handled as per `micr_scan_instrument_number`; output must be a contiguous 9-digit string.\n"
        "**Extraction Method - CRITICAL STEPS & ROBUSTNESS:**\n"
        "  1.  **Sequential Logic:** This extraction strictly depends on the successful prior identification of `micr_scan_instrument_number` and its trailing delimiter.\n"
        "  2.  **E-13B OCR:** Apply E-13B specialized OCR to the segment following the identified `micr_scan_instrument_number`.\n"
        "  3.  **Targeted Pattern Match:** Locate the 9-digit sequence matching the `⑈DDDDDDDDD⑆` pattern in the expected position.\n"
        "  4.  **Aggressive Digit Filtering:** Extract **ONLY the 9 numeric E-13B digits**. Exclude delimiters (`⑈`, `⑆`) and any other non-digits.\n"
        "  5.  **Delimiter Integrity Check:** If delimiters are imperfectly recognized but a clear 9-digit E-13B sequence is present in the correct position relative to the instrument number, extract with lowered confidence and justification (e.g., 'Imperfect leading ⑈ or trailing ⑆ for payee_details'). If ambiguity about the segment's boundaries or content arises due to faulty delimiters, extraction quality is compromised.\n"
        "  6.  **Indian Context Validation (CCCBBBAAA):** The 9-digit code typically follows a City (3), Bank (3), Branch (3) structure. This can be a soft validation. However, direct OCR of 9 clear E-13B digits as per pattern is the primary driver.\n"
        "  7.  **Print Quality Handling:** Apply same principles as for `micr_scan_instrument_number` regarding smudged/broken E-13B digits.\n"
        "**Error Handling:** If a clear 9-digit E-13B sequence matching the expected `⑈DDDDDDDDD⑆` pattern and location cannot be confidently identified, this field must be null, with confidence < 0.5 and reason 'Sort code segment (payee_details) not found or illegible'.\n"
        "**Output:** A string containing exactly 9 numeric digits. Null if criteria not met."
    ),
    "micr_scan_micr_acno": (
        "**Objective:** Extract a 6-digit account-related or secondary transaction code from the E-13B MICR line, if structurally present.\n"
        "**Overall MICR Structure Expectation:** This field, if present, is the **third segment** in the MICR line, following `micr_scan_payee_details`. Its presence is conditional based on the cheque's MICR layout.\n"
        "**Positional Definition:** This numeric group, **if present**, **must immediately follow the complete `micr_scan_payee_details` (9-digit sort code) segment** (i.e., after the `micr_scan_payee_details`' trailing `⑆` symbol).\n"
        "**Pattern and Delimiters (Strict Interpretation & Conditional Presence):**\n"
        "  * If this segment exists, it consists of **exactly 6 numeric E-13B digits (0-9)**.\n"
        "  * This sequence, when present, is **critically expected to be enclosed by a leading MICR 'Transit' symbol (⑆) and a trailing MICR 'On-Us' symbol (⑈)**, forming the pattern: `⑆DDDDDD⑈`.\n"
        "  * **Preceded by:** The complete `micr_scan_payee_details` segment (which ends with `⑆`). If present, this `micr_scan_micr_acno` segment starts with its own `⑆` delimiter.\n"
        "  * **Followed by:** If present, the `micr_scan_instrument_type` segment. The end of the `micr_scan_micr_acno` segment is defined by its trailing `⑈` symbol.\n"
        "**Extraction Method - CRITICAL STEPS & ROBUSTNESS:**\n"
        "  1.  **Sequential Logic:** Extraction depends on successful prior identification of `micr_scan_payee_details` and its trailing delimiter.\n"
        "  2.  **E-13B OCR:** Apply E-13B specialized OCR to the segment following the identified `micr_scan_payee_details`.\n"
        "  3.  **Pattern Match & Presence Check:** Attempt to locate a 6-digit E-13B sequence matching the `⑆DDDDDD⑈` pattern in the expected position.\n"
        "  4.  **Aggressive Digit Filtering (if present):** If found, extract **ONLY the 6 numeric E-13B digits**. Exclude delimiters (`⑆`, `⑈`) and non-digits.\n"
        "  5.  **Handling Structural Absence (Critical):** If the MICR line structure indicates this 6-digit segment (as defined by the `⑆DDDDDD⑈` pattern) is **not present** between the `micr_scan_payee_details` (ending in `⑆`) and the `micr_scan_instrument_type` (or end of MICR line), the value for `micr_scan_micr_acno` **MUST be '000000'**. The confidence for this default value should be high (e.g., 0.95) if absence is clear from structure, with reason 'micr_acno segment structurally absent, default applied'.\n"
        "  6.  **Delimiter Integrity & Ambiguity:** If delimiters are imperfect but a 6-digit E-13B sequence is plausible in this position, extract with reduced confidence and justification. If the segment is unclear, or if digits are present but don't match the 6-digit length within expected delimiters (e.g., `⑆DDDDD⑈` or `⑆DDDDDDD⑈`), this specific field definition is not met. It should then be treated as structurally absent ('000000') or, if severely garbled, null with very low confidence and reason for ambiguity (e.g., 'micr_acno segment present but malformed').\n"
        "  7.  **Print Quality Handling:** Apply same principles for smudged/broken E-13B digits if the segment is deemed present and not structurally absent.\n"
        "**Error Handling:** If the segment is deemed present (i.e., not structurally absent) but is illegible or doesn't conform to the 6-digit requirement within its delimiters, set to null with confidence < 0.5 and reason 'micr_acno segment present but illegible/non-conforming'. If structurally absent, use '000000' as specified.\n"
        "**Output:** A string containing exactly 6 numeric digits if present and correctly parsed, or '000000' if structurally absent as per rules. Null for actual read errors of a present field that cannot be confidently extracted."
    ),
    "micr_scan_instrument_type": (
        "**Objective:** Extract the 2-digit transaction code or instrument type from the very end of the E-13B MICR line.\n"
        "**Overall MICR Structure Expectation:** This field is the **fourth and typically final segment** in the MICR line.\n"
        "**Positional Definition:** This is **strictly the last numeric group**, consisting of 2 E-13B digits, found at the **terminal end** of the MICR encoding sequence.\n"
        "**Pattern and Delimiters (Strict Interpretation):**\n"
        "  * Search for a sequence of **exactly 2 numeric E-13B digits (0-9)**.\n"
        "  * These digits are at the terminal end of the recognizable MICR character sequence. They are often visually separated by a larger space from any preceding MICR symbols or numbers, especially if `micr_scan_micr_acno` is absent. No specific trailing delimiter is expected after these two digits other than the end of the scannable MICR zone.\n"
        "  * **Preceded by:**\n"
        "      * If `micr_scan_micr_acno` is present and parsed (ending with pattern `⑆DDDDDD⑈`): This 2-digit `micr_scan_instrument_type` segment will appear immediately after the trailing `⑈` of `micr_scan_micr_acno`.\n"
        "      * If `micr_scan_micr_acno` is structurally absent (or defaulted to '000000'): This 2-digit `micr_scan_instrument_type` segment will appear after the trailing `⑆` of the `micr_scan_payee_details` segment (pattern `⑈DDDDDDDDD⑆`). There might be a noticeable space between the `⑆` and these two digits.\n"
        "  * **Followed by:** Nothing (end of the MICR scannable zone/clear band).\n"
        "  * Common Indian instrument types include '10' (Savings), '11' (Current), '29' (Govt.), '31' (CTS Standard), etc.\n"
        "**Extraction Method - CRITICAL STEPS & ROBUSTNESS:**\n"
        "  1.  **Sequential Logic & End-of-Line Focus:** After processing all preceding MICR segments (`micr_scan_instrument_number`, `micr_scan_payee_details`, and conditional `micr_scan_micr_acno`), specifically target the terminal characters of the MICR line.\n"
        "  2.  **E-13B OCR:** Apply E-13B specialized OCR to the identified terminal region.\n"
        "  3.  **Isolate Final Two Digits:** Identify the final two clearly recognizable E-13B numeric digits. These should be the absolute last digits before the MICR clear band ends or non-MICR print/paper edge is encountered, considering the preceding field's ending delimiter.\n"
        "  4.  **Aggressive Digit Filtering:** Extract **ONLY the 2 numeric E-13B digits**. Exclude any preceding symbols (which properly belong to previous fields if not already accounted for) or surrounding spaces.\n"
        "  5.  **Ambiguity at End-of-Line:** If the MICR line ends with unclear characters, or more than two digits without clear segmentation from a preceding valid field (e.g., if the sequence is `...⑈XXYY` where `XX` are the target digits, it is clear; but if it's `...⑆XXXXYY` and `XXXX` was not a valid `micr_scan_micr_acno`, this could be ambiguous for a 2-digit field if `XX` are also digits), extraction may fail or have low confidence. The system must be certain these are the *intended final two distinct digits* for this code.\n"
        "  6.  **Print Quality Handling:** Apply same principles for smudged/broken E-13B digits.\n"
        "**Error Handling:** If a clear 2-digit E-13B sequence cannot be confidently identified at the end of the MICR line in the expected position relative to prior fields, this field must be null, with confidence < 0.5 and reason 'Instrument type segment not found or illegible at end of MICR'.\n"
        "**Output:** A string containing exactly 2 numeric digits. Null if criteria not met."
    ),
    "signature_present": (
        "**Objective:** Determine if a handwritten signature exists in the designated area.\n"
        "**Primary Location Strategy:** Analyze the designated signature space, typically **bottom-right**, above the `issuer_name` (if present) and MICR line.\n"
        "**Method:** Detect the presence of connected, free-flowing ink strokes characteristic of a handwritten signature. **Do NOT attempt to read or validate the signature's authenticity.** Distinguish from printed text, stamps, or incidental marks.\n"
        "**Output:** **Strictly 'YES' or 'NO'.**"
    ),
    "IFSC": (
        "**Objective:** Extract the 11-character Indian Financial System Code.\n"
        "**Primary Location Strategy:** Search near the `bank_name` and `bank_branch` details. Look for explicit labels: 'IFSC', 'IFS Code', 'IFSC Code'.\n"
        "**Format:** **Strictly 11 alphanumeric characters.**\n"
        "  - Format: **AAAA0XXXXXX**\n"
        "  - First 4 characters: Alphabetic (Bank Code)\n"
        "  - 5th character: MUST BE ZERO ('0')\n"
        "  - Last 6 characters: Alphanumeric (Branch Code)\n"
        "**Extraction Method:**\n"
        "  1. Scan the target area for strings matching the 11-character pattern.\n"
        "  2. Apply robust character recognition, paying attention to common confusions (0/O, 1/I, S/5, B/8).\n"
        "  3. **VALIDATE RIGOROUSLY:**\n"
        "     a. Check total length is exactly 11.\n"
        "     b. Verify the 5th character is '0'.\n"
        "     c. Verify the first 4 characters are alphabetic.\n"
        "  4. Cross-reference the first 4 characters with the extracted `bank_name`'s expected code if possible.\n"
        "  5. Handle specific layouts for non-standard cheque types (Drafts, Manager's Cheques) where IFSC might be positioned differently.\n"
        "**Output:** The validated 11-character IFSC code. If validation fails, output 'Error' / 'Not Found'."
    ),
    "currency": (
        "**Objective:** Identify the currency of the transaction.\n"
        "**Extraction Method (Prioritized):**\n"
        "  1. **Explicit Code:** Look for printed ISO 4217 codes (e.g., 'INR', 'USD', 'EUR') on the cheque body.\n"
        "  2. **Numeric Symbol:** Identify currency symbols adjacent to `amount_numeric` (e.g., '₹', '$', '£', '€'). Normalize '₹' and text 'INR' as Indian Rupee.\n"
        "  3. **Words Amount:** Extract currency words from `amount_words` (e.g., 'Rupees', 'Dollars', 'रुपये'). Handle multilingual terms.\n"
        "  4. **Contextual Default:** If an Indian bank (`bank_name`, `IFSC`) is identified and no other currency indicators are present, default to 'INR'.\n"
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