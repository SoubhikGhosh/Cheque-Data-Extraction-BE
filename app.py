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
    version="1.0.0"
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
    def process_multimodal_document(file_data: bytes, file_type: str, file_path: str) -> Dict[str, Any]:
        """Process a cheque document using Vertex AI's multimodal capabilities."""
        try:
            # Initialize Vertex AI model
            model = GenerativeModel("gemini-1.5-flash-002", safety_settings=safety_settings)
            
            result = {}
            
            # Create a Vertex AI Part from the file data
            file_part = Part.from_data(data=file_data, mime_type=file_type)
            
            # For images, process directly
            if file_type.lower() in ["image/jpeg", "image/jpg", "image/png", "image/tiff"]:
                
                # Now extract text and fields based on the document type
                doc_fields = [field['name'] for field in FIELDS]
                fields_str = ", ".join(doc_fields)

                field_descriptions = {
                    "bank_name": "The name of the bank issuing the cheque, always on the top left of the cheque, in case there are multiple ALWAYS go for the one on the top left corner. Use contextual understanding about major Indian and foreign banks and pattern recognition to correctly approximate if the image is unclear.",
                    "bank_branch": "The specific branch of the bank where the account is maintained, generally just below the bank name",
                    "account_number": "The bank account number from which the cheque is drawn, also denoted by A/C No.",
                    "date": "The date when the cheque was issued (in YYYY-MM-DD format), generally on the top right",
                    "payee_name": "The name of the person or entity to whom the cheque is payable, generally handwritten, below the Bank Name and details after the word PAY or pay or Pay",
                    "amount_words": "The amount written in words on the cheque, generally handwritten, below payee name, maybe spread over 2 lines",
                    "amount_numeric": "The amount written in numeric digits on the cheque, generally handwritten in the right middle section",
                    "currency": "The currency of the cheque (e.g., INR, USD, EUR), may also have to classify basis currency sign",
                    "issuer_name": "The name of the person or entity issuing the cheque, always just below or above the signature",
                    "micr_code": "Magnetic Ink Character Recognition code printed on the bottom of the cheque, always printed",
                    "signature_present": "Whether a signature is present on the cheque, always handwritten ink free flowing strokes on the bottom right area above micr code. use only YES or NO.",
                    "IFSC": "The IFS Code assigned to a particular branch of a bank where the cheque was issued from, denoted by IFS Code or IFSC or IFSC Code"
                }

                # Create field list with descriptions
                fields_with_descriptions = []
                for field in doc_fields:
                    description = field_descriptions.get(field, "")
                    fields_with_descriptions.append(f"- {field}: {description}")
                
                fields_list = "\n".join(fields_with_descriptions)
  
                extraction_prompt = f"""
                You are a specialized financial document analyzer with expertise in extracting information from scanned cheque documents.
            
                Extract the following fields from the cheque with maximum precision:
                {fields_list}
                
                For each field:
                1. Extract the exact value as it appears in the document
                2. If the text is unclear, make a reasonable approximation
                3. For dates, standardize to YYYY-MM-DD format
                4. For monetary amounts, include both value and currency
                5. Assign a confidence score between 0.0 and 1.0 for each extraction
                6. If a field cannot be found or extracted, provide a reason why
                
                For each field, provide:
                - The extracted value
                - A confidence score (0.0-1.0)
                - The exact text segment from which you extracted the information
                - A reason if the field couldn't be extracted
                
                Format your response as a JSON with two parts:
                1. "full_text": The complete text from the cheque
                2. "extracted_fields": An array of objects with field_name, value, and confidence
                
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
                             
            else:
                logger.warning(f"Unsupported file type for Vertex AI processing: {file_type}")
                result = {
                    "error": f"Unsupported file type: {file_type}",
                    "text": "",
                    "pages": [],
                    "extracted_fields": []
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
                "extracted_fields": []
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

    # ============ NEW BATCH PROCESSING METHOD ============
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

# ============ OPTIMIZED ZIP FILE PROCESSING FUNCTION ============
def process_zip_files(file_contents: List[bytes], file_names: List[str], job_id: str):
    """Process multiple zip files and generate Excel report using Gemini's multimodal capabilities with parallel processing."""

    logger.info(f"Starting optimized process_zip_files for job {job_id}")
    logger.info(f"Number of files: {len(file_contents)}")
    logger.info(f"File names: {file_names}")

    try:
        # Create a temp directory for this job
        temp_dir = tempfile.mkdtemp(prefix=f"job_{job_id}_")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Dictionary to store results for each folder
        folder_results = {}
        total_files = 0
        processed_files = 0
        
        # For each zip file
        for zip_index, (zip_content, zip_name) in enumerate(zip(file_contents, file_names)):
            # Extract the zip file to temp directory
            zip_dir = os.path.join(temp_dir, os.path.splitext(zip_name)[0])
            os.makedirs(zip_dir, exist_ok=True)
            
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                zf.extractall(zip_dir)
            
            # Structure to organize files by folder for batch processing
            folder_files = {}
            
            # First pass: gather files by folder
            for root, dirs, files in os.walk(zip_dir):
                # Skip the root directory
                if root == zip_dir:
                    continue
                
                # Get folder name (relative to zip)
                rel_path = os.path.relpath(root, zip_dir)
                folder_name = rel_path
                
                # Skip if there are no files
                if not files:
                    continue
                
                # Initialize folder results if not already present
                if folder_name not in folder_results:
                    folder_results[folder_name] = []
                
                # Initialize folder files
                if folder_name not in folder_files:
                    folder_files[folder_name] = []
                
                # Add files to the folder's file list
                for file in files:
                    if file.startswith('.') or file.startswith('~'):
                        continue  # Skip hidden files
                    
                    file_path = os.path.join(root, file)
                    total_files += 1
                    
                    # Get file extension
                    _, ext = os.path.splitext(file)
                    
                    # Only include supported file types
                    if ext.lower() not in ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                        continue
                    
                    # Determine file type
                    file_type = {
                        '.pdf': 'application/pdf',
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                        '.tiff': 'image/tiff',
                        '.tif': 'image/tiff'
                    }.get(ext.lower(), 'application/octet-stream')
                    
                    # Read file data
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                    
                    # Add to folder files for batch processing
                    folder_files[folder_name].append({
                        'path': file_path,
                        'data': file_data,
                        'type': file_type
                    })
            
            # Second pass: Process files in batches by folder
            for folder_name, files_list in folder_files.items():
                logger.info(f"Processing folder {folder_name} with {len(files_list)} files")
                
                # Process in batches
                for i in range(0, len(files_list), BATCH_SIZE):
                    batch = files_list[i:i+BATCH_SIZE]
                    logger.info(f"Processing batch {i//BATCH_SIZE + 1} with {len(batch)} files")
                    
                    # Process the batch
                    batch_results = ChequeProcessor.process_document_batch(batch)
                    
                    # Process batch results
                    for result in batch_results:
                        file_path = result.get('file_path', '')
                        
                        # Update folder results with extracted fields
                        for field in result.get("extracted_fields", []):
                            field_info = {
                                "field_name": field.get("field_name", ""),
                                "value": field.get("value", ""),
                                "confidence": field.get("confidence", 0.0),
                                "reason": field.get("reason", "")
                            }
                            
                            # Add to folder results
                            folder_results[folder_name].append({
                                "filepath": file_path,
                                "field_name": field_info["field_name"],
                                "value": field_info["value"],
                                "confidence": field_info["confidence"],
                                "reason": field_info["reason"]
                            })
                        
                        processed_files += 1
                        
                        # Update progress periodically
                        if processed_files % 10 == 0:
                            elapsed_time = time.time() - processed_jobs[job_id]["start_time"]
                            if processed_files > 0 and total_files > 0:
                                files_per_second = processed_files / elapsed_time
                                remaining_files = total_files - processed_files
                                estimated_time_remaining = remaining_files / files_per_second if files_per_second > 0 else 0
                                
                                # Format time remaining into hours, minutes, seconds
                                hours, remainder = divmod(estimated_time_remaining, 3600)
                                minutes, seconds = divmod(remainder, 60)
                                time_format = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                                
                                logger.info(f"Progress: {processed_files}/{total_files} files processed "
                                        f"({processed_files/total_files*100:.1f}%). "
                                        f"Processing rate: {files_per_second:.2f} files/sec. "
                                        f"Estimated time remaining: {time_format}")
                                
                                # Update job status with progress information
                                processed_jobs[job_id].update({
                                    "processed_files": processed_files,
                                    "processing_rate": files_per_second,
                                    "estimated_time_remaining": estimated_time_remaining
                                })

        # Generate Excel report
        excel_path = os.path.join(output_dir, f"cheque_extraction_results_{job_id}.xlsx")
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            # Process each folder's results
            for folder_name, results in folder_results.items():
                if not results:
                    continue
                    
                # Create DataFrame
                df = pd.DataFrame()
                
                # Group by filepath
                filepath_groups = {}
                
                for item in results:
                    filepath = item["filepath"]
                    if filepath not in filepath_groups:
                        filepath_groups[filepath] = {
                            "filepath": filepath,
                        }
                    
                    # Add field name, value, confidence
                    if "field_name" in item:
                        field_name = item["field_name"]
                        filepath_groups[filepath][field_name] = item["value"]
                        filepath_groups[filepath][f"{field_name}_conf"] = item["confidence"]
                        
                        if item.get("reason"):
                            filepath_groups[filepath][f"{field_name}_reason"] = item["reason"]
                
                # Convert to DataFrame
                if filepath_groups:
                    df = pd.DataFrame(list(filepath_groups.values()))
                    
                    # Reorder columns to put field and confidence side by side
                    cols = ["filepath"]
                    for field in FIELDS:
                        field_name = field["name"]
                        if field_name in df.columns:
                            cols.append(field_name)
                            cols.append(f"{field_name}_conf")
                            if f"{field_name}_reason" in df.columns:
                                cols.append(f"{field_name}_reason")
                    
                    # Use only columns that exist in the DataFrame
                    cols = [col for col in cols if col in df.columns]
                    if cols:  # Only reindex if columns exist
                        df = df[cols]
                
                # Create a sanitized sheet name
                sheet_name = re.sub(r'[\\/*?[\]:]', '_', folder_name)
                if len(sheet_name) > 31:
                    sheet_name = sheet_name[:28] + '...'
                
                # Write to Excel
                if not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Update job status in-memory
        processed_jobs[job_id] = {
            "status": "completed",
            "start_time": time.time(),
            "end_time": time.time(),
            "total_files": total_files,
            "processed_files": processed_files,
            "output_file_path": excel_path
        }
        
        logger.info(f"Job {job_id} completed. Output file: {excel_path}")
        return excel_path
        
    except Exception as e:
        logger.error(f"Error processing zip files: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Update job status to failed
        processed_jobs[job_id] = {
            "status": "failed",
            "start_time": time.time(),
            "end_time": time.time(),
            "total_files": total_files if 'total_files' in locals() else 0,
            "processed_files": processed_files if 'processed_files' in locals() else 0,
            "error_message": str(e)
        }

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