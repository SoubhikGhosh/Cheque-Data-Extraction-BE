import uuid
import time
import logging
import traceback # For detailed exception logging in task wrapper
from typing import List, Tuple

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse

from app.core.config import settings
from app.services.vertex_ai_service import VertexAIService
from app.services.file_service import FileService
from app.services.cheque_processing_service import ChequeProcessingService
from app.api.v1.schemas import FileUploadResponse, JobStatusDetail
# from app.main import app_executor # If executor is managed globally in main.py

logger = logging.getLogger(__name__)

router = APIRouter()

# --- Dependency Injection for Services ---
# This makes services easily mockable for tests and manages their lifecycle if needed.
def get_vertex_ai_service():
    return VertexAIService()

def get_file_service():
    return FileService()

def get_cheque_processing_service(
    vertex_service: VertexAIService = Depends(get_vertex_ai_service),
    file_service: FileService = Depends(get_file_service)
):
    return ChequeProcessingService(vertex_service, file_service)
# --- End Dependency Injection ---


@router.post("/upload", response_model=FileUploadResponse)
async def upload_files_for_processing(
    files: List[UploadFile] = File(...),
    cheque_processor: ChequeProcessingService = Depends(get_cheque_processing_service),
    file_service: FileService = Depends(get_file_service) # For cleaning temp dir in wrapper
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    
    uploaded_file_data: List[Tuple[bytes, str]] = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith('.zip'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type for {file.filename}. Only ZIP files are accepted."
            )
        try:
            content = await file.read() # Read file content
            uploaded_file_data.append((content, file.filename))
        except Exception as e:
            logger.error(f"Failed to read uploaded file {file.filename}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error reading file: {file.filename}")
        finally:
            await file.close()


    job_id = str(uuid.uuid4())

    # Wrapper for the background task
    def task_wrapper():
        base_temp_dir_for_job = "" # To store path for cleanup
        try:
            # The process_uploaded_zip_files will manage its own temp subdirectories
            # but we might need the root job temp dir for final cleanup
            base_temp_dir_for_job = file_service.create_temp_dir(f"task_wrapper_{job_id}") # Or get from service if it returns it
            
            # Call the main processing function from the service
            cheque_processor.process_uploaded_zip_files(job_id, uploaded_file_data)
            
            logger.info(f"Background task for job {job_id} completed wrapper successfully.")
        except Exception as e_task:
            logger.error(f"Background task for job {job_id} failed: {e_task}", exc_info=True)
            # Update job status to failed if not already handled by the service's own try-except
            # The service should ideally handle its own status updates comprehensively.
            # This is a fallback.
            job_details = cheque_processor.get_processed_job_detail(job_id)
            if job_details and job_details.status != "failed":
                 cheque_processor.update_job_details(job_id, {
                    "status": "failed",
                    "error_message": f"Task wrapper error: {str(e_task)}",
                    "error_traceback": traceback.format_exc(),
                    "end_time": time.time()
                })
        finally:
            # Clean up the main temporary directory created by the service or this wrapper
            # The service's process_uploaded_zip_files creates a base_temp_dir.
            # That path should be cleaned.
            # Assuming process_uploaded_zip_files either cleans up or its temp path is known.
            # For robustness, the service itself should handle its temp dir cleanup.
            # If `base_temp_dir_for_job` was used by the service, it would be cleaned here.
            # Let's assume the service's `process_uploaded_zip_files` manages its own temp dir's lifecycle.
            # If the temp dir is created *inside* process_uploaded_zip_files, it should be cleaned there.
            # The `base_temp_dir` in `process_uploaded_zip_files` is the one to clean.
            # This requires careful coordination or the service cleaning up itself.
            # For now, let's assume the *service* is responsible for cleaning its primary temp dir.
            # This wrapper's `base_temp_dir_for_job` might be redundant if service manages all.
            if base_temp_dir_for_job and os.path.exists(base_temp_dir_for_job): # If this wrapper created one
                file_service.clean_temp_dir(base_temp_dir_for_job)

            cheque_processor.remove_active_task(job_id) # Remove from active tasks dict

    # Submit the task to the executor managed by the service or a global one
    future = cheque_processor.executor.submit(task_wrapper)
    cheque_processor.add_active_task(job_id, future)

    # Initial job status update (minimal, more details set by the task itself)
    # The service's process_uploaded_zip_files will set a more detailed initial status.
    # This is just to ensure _processed_jobs has an entry immediately.
    if not cheque_processor.get_processed_job_detail(job_id): # If not already initialized by the service start
        initial_status = JobStatusDetail(
            job_id=job_id,
            status="queued", # or "initializing"
            start_time=time.time(),
            input_files=[name for _, name in uploaded_file_data]
        )
        # This direct update to _processed_jobs is not ideal if service owns it.
        # Better: cheque_processor.initialize_job_status(job_id, ...)
        # For now, rely on service's initialization in process_uploaded_zip_files
        # _processed_jobs[job_id] = initial_status

    return FileUploadResponse(
        status="processing_initiated",
        job_id=job_id,
        message="Cheque extraction job initiated. Check status endpoint.",
        files=[file_tuple[1] for file_tuple in uploaded_file_data],
        timestamp=int(time.time())
    )


@router.get("/status/{job_id}", response_model=JobStatusDetail)
async def check_job_status(job_id: str, cheque_processor: ChequeProcessingService = Depends(get_cheque_processing_service)):
    job_detail = cheque_processor.get_job_status(job_id)
    if not job_detail:
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found.")
    return job_detail


@router.get("/download/{job_id}")
async def download_results(job_id: str, cheque_processor: ChequeProcessingService = Depends(get_cheque_processing_service)):
    job_detail = cheque_processor.get_job_status(job_id) # Use get_job_status to ensure active tasks are checked
    
    if not job_detail:
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found.")
    
    if job_detail.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed. Current status: {job_detail.status}. "
                   f"Error: {job_detail.error_message or 'None'}"
        )
    
    output_path = job_detail.output_file_path
    if not output_path or not os.path.exists(output_path):
        logger.error(f"Output file path not found or invalid for completed job {job_id}: {output_path}")
        raise HTTPException(status_code=404, detail=f"Output file for job {job_id} not found on server.")
    
    file_name = os.path.basename(output_path) # Get filename from the full path

    return FileResponse(
        path=output_path,
        filename=file_name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )