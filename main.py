# /cheque_extraction_api/main.py

import logging
import uuid
import time
import os
import traceback
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import vertexai

import config
import utils
from processing import process_zip_file_and_generate_report

# --- Initialization ---
utils.configure_logging()
logger = logging.getLogger(__name__)

try:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    logger.info(f"Ensured output directory exists at: ./{config.OUTPUT_DIR}")
except OSError as e:
    logger.error(f"Fatal Error: Could not create output directory ./{config.OUTPUT_DIR}. Error: {e}")


# Initialize Vertex AI
try:
    vertexai.init(
        project=config.GCP_PROJECT_ID,
        location=config.GCP_LOCATION,
        api_endpoint=config.API_ENDPOINT
    )
    logger.info("Vertex AI initialized successfully.")
except Exception as e:
    logger.error(f"Fatal Error: Could not initialize Vertex AI. {e}")
    # Exit or handle appropriately if Vertex AI is essential at startup
    # For now, we log and continue, but endpoints will fail.

# --- App & State Management ---
app = FastAPI(
    title="Cheque Data Extraction API",
    description="API for processing zip files of cheque images using Vertex AI.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for job statuses.
# For production, consider using Redis or a database.
processed_jobs: Dict[str, Dict] = {}
# Global thread pool for handling background processing jobs
executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS, thread_name_prefix='JobHandler')


# --- API Endpoints ---
@app.post("/upload")
async def upload_files_for_processing(files: List[UploadFile] = File(...)):
    """
    Accepts one or more ZIP files, assigns a job ID, and starts
    the extraction process in the background.
    """
    job_id = str(uuid.uuid4())
    logger.info(f"Received new job with ID: {job_id}")

    file_contents = []
    file_names = []
    for file in files:
        if not file.filename.lower().endswith('.zip'):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}. Only .zip files are accepted.")
        file_contents.append(await file.read())
        file_names.append(file.filename)
        logger.info(f"Job {job_id}: Staged file '{file.filename}' for processing.")

    # Initialize job status
    processed_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "start_time": time.time(),
        "input_files": file_names,
        "total_files": 0,
        "processed_files": 0,
        "progress_percentage": 0.0,
    }

    # Submit the long-running task to the thread pool
    executor.submit(
        process_zip_file_and_generate_report,
        job_id,
        file_contents,
        file_names,
        processed_jobs[job_id] # Pass the dictionary to be updated by the thread
    )

    return {
        "message": "Job successfully queued for processing.",
        "job_id": job_id,
        "status_endpoint": f"/status/{job_id}"
    }

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Returns the current status of a processing job.
    """
    job = processed_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job

@app.get("/download/{job_id}")
async def download_result_file(job_id: str):
    """
    Allows downloading of the generated Excel report for a completed job.
    """
    job = processed_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not complete. Current status: {job.get('status')}")

    output_path = job.get("output_file_path")
    if not output_path or not os.path.exists(output_path):
        logger.error(f"File not found for job {job_id} at path: {output_path}")
        raise HTTPException(status_code=404, detail="Output file not found on server.")

    return FileResponse(
        path=output_path,
        filename=os.path.basename(output_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Cheque Extraction API is running."}

# --- Cleanup ---
@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down thread pool executor.")
    executor.shutdown(wait=True)