import os
import re
import json
import time
import logging
import traceback
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from pydantic import ValidationError


from app.core.config import settings
from app.constants import FIELDS, DEFAULT_ERROR_SIGNATURE_COORDS, EXCEL_FILENAME_TEMPLATE, EXCEL_SHEET_NAME_MAX_LEN
from app.prompts.cheque_prompts import get_cheque_extraction_prompt, SIGNATURE_EXTRACTION_PROMPT
from app.services.vertex_ai_service import VertexAIService
from app.services.file_service import FileService
from app.api.v1.schemas import (
    MultimodalProcessingResult, 
    SignatureExtractionResult, 
    ExtractedFieldData,
    LLMExtractionOutput,
    ExcelRow,
    JobStatusDetail
)
# from app.utils.image_utils import crop_image_with_coordinates # Not used as per new requirement (no image cropping for excel)

logger = logging.getLogger(__name__)

# In-memory store for job tracking.
# For production, replace with a persistent store (e.g., Redis, DB).
_active_tasks: Dict[str, concurrent.futures.Future] = {}
_processed_jobs: Dict[str, JobStatusDetail] = {}


class ChequeProcessingService:
    def __init__(self, vertex_ai_service: VertexAIService, file_service: FileService):
        self.vertex_ai_service = vertex_ai_service
        self.file_service = file_service
        # Consider initializing executor here if it's to be managed by this service instance
        # Or it can be a global executor passed around or accessed.
        # For now, creating it when needed in process_document_batch or making it module level.
        # Making it module level as per original design for ThreadPoolExecutor
        global_max_workers = settings.MAX_WORKERS
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=global_max_workers)
        logger.info(f"ThreadPoolExecutor initialized with max_workers={global_max_workers}")


    def _process_single_document(
        self, file_data: bytes, file_mime_type: str, file_path_for_logging: str
    ) -> MultimodalProcessingResult:
        """
        Processes a single document (image) using Vertex AI for field extraction and signature detection.
        """
        try:
            # 1. Extract main cheque data
            extraction_prompt_str = get_cheque_extraction_prompt()
            llm_output: Optional[LLMExtractionOutput] = self.vertex_ai_service.extract_data_from_document(
                file_data=file_data,
                file_mime_type=file_mime_type,
                prompt=extraction_prompt_str,
                file_path_for_logging=file_path_for_logging
            )

            extracted_fields_list = []
            full_text_content = ""

            if llm_output:
                full_text_content = llm_output.full_text or ""
                extracted_fields_list = llm_output.extracted_fields
            else: # Handle case where llm_output is None (major error in VertexAIService call)
                 logger.error(f"LLM output was None for {file_path_for_logging}. No fields extracted.")
                 # Create a minimal error result
                 return MultimodalProcessingResult(
                    file_path=file_path_for_logging,
                    text="Failed to get LLM response",
                    extracted_fields=[],
                    signature_coordinates=SignatureExtractionResult(**DEFAULT_ERROR_SIGNATURE_COORDS),
                    error="LLM processing failed"
                 )


            # 2. Extract signature coordinates
            signature_result: SignatureExtractionResult = self.vertex_ai_service.extract_signature_coordinates(
                file_data=file_data,
                file_mime_type=file_mime_type,
                prompt=SIGNATURE_EXTRACTION_PROMPT,
                file_path_for_logging=file_path_for_logging
            )

            # 3. Combine results
            # Append signature coordinates as a special field if it exists
            if signature_result.exists and signature_result.coordinates:
                try:
                    # Ensure signature_coordinates value is a JSON string as expected by original logic for DataFrame
                    coords_json_str = signature_result.coordinates.model_dump_json()
                    sig_field_data = ExtractedFieldData(
                        field_name="signature_coordinates",
                        value=coords_json_str, # Store as JSON string
                        confidence=signature_result.confidence,
                        reason=signature_result.description
                    )
                    extracted_fields_list.append(sig_field_data)
                except Exception as e: # Catch potential model_dump_json error
                    logger.error(f"Error serializing signature coordinates for {file_path_for_logging}: {e}")
                    extracted_fields_list.append(ExtractedFieldData(
                        field_name="signature_coordinates",
                        value="Error serializing coordinates",
                        confidence=0.0,
                        reason=str(e)
                    ))


            return MultimodalProcessingResult(
                file_path=file_path_for_logging, # Store the original path for tracking
                text=full_text_content,
                # Assuming single-page images for cheques, so pages structure is simplified
                pages=[{"page_num": 1, "text": full_text_content}], 
                extracted_fields=extracted_fields_list,
                signature_coordinates=signature_result
            )

        except Exception as e:
            logger.error(f"Unhandled error processing document {file_path_for_logging}: {e}", exc_info=True)
            return MultimodalProcessingResult(
                file_path=file_path_for_logging,
                error=f"General processing error: {str(e)}",
                text="",
                extracted_fields=[],
                signature_coordinates=SignatureExtractionResult(**DEFAULT_ERROR_SIGNATURE_COORDS)
            )


    def _process_document_batch_concurrently(
        self, file_batch: List[Dict[str, Any]]
    ) -> List[MultimodalProcessingResult]:
        """
        Processes a batch of documents in parallel using the class's ThreadPoolExecutor.
        file_batch: list of dictionaries, each with 'data', 'type', 'path' (original path for logging).
        """
        batch_results: List[MultimodalProcessingResult] = []
        
        # Use a local executor for the batch, or the class's shared executor
        # If using class executor, ensure it's robust for many calls.
        # Max workers for this specific batch call can be limited by batch size or global max_workers
        # executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(len(file_batch), settings.MAX_WORKERS))
        
        futures_map: Dict[concurrent.futures.Future, Dict[str, Any]] = {}
        for file_info in file_batch:
            # 'path' here is the original_filename or full path for logging purposes.
            # 'data' is file_data (bytes), 'type' is mime_type.
            future = self.executor.submit(
                self._process_single_document,
                file_info['data'],
                file_info['type'],
                file_info['path'] # This path is for logging in _process_single_document
            )
            futures_map[future] = file_info

        for future in concurrent.futures.as_completed(futures_map):
            file_info_processed = futures_map[future]
            original_file_path_logged = file_info_processed['path'] # Path used in logging for this file
            try:
                result: MultimodalProcessingResult = future.result()
                # Ensure file_path in the result is the one we expect for aggregation
                result.file_path = original_file_path_logged # Overwrite if necessary, should match
                batch_results.append(result)
                logger.info(f"Successfully processed (in batch): {original_file_path_logged}")
            except Exception as e:
                logger.error(f"Error processing file {original_file_path_logged} within batch: {e}", exc_info=True)
                batch_results.append(MultimodalProcessingResult(
                    file_path=original_file_path_logged,
                    error=f"Batch processing future error: {str(e)}",
                    text="",
                    extracted_fields=[],
                    signature_coordinates=SignatureExtractionResult(**DEFAULT_ERROR_SIGNATURE_COORDS)
                ))
        return batch_results


    def process_uploaded_zip_files(self, job_id: str, uploaded_files: List[Tuple[bytes, str]]):
        """
        Main function to process uploaded zip files, extract cheque data, and generate an Excel report.
        Manages job status via global dictionaries.
        uploaded_files: List of (file_content_bytes, file_name_str)
        """
        job_start_time = time.time()
        # Initialize job status
        _processed_jobs[job_id] = JobStatusDetail(
            job_id=job_id,
            status="processing",
            start_time=job_start_time,
            input_files=[name for _, name in uploaded_files],
            total_files=0, # Will be updated
            processed_files=0,
            progress_percentage=0.0
        )
        logger.info(f"Starting job {job_id} with {len(uploaded_files)} ZIP file(s).")

        base_temp_dir = self.file_service.create_temp_dir(job_id)
        output_excel_path = os.path.join(base_temp_dir, "output") # Store excel in an output subfolder
        os.makedirs(output_excel_path, exist_ok=True)
        excel_file_name = EXCEL_FILENAME_TEMPLATE.format(job_id=job_id)
        final_excel_file_path = os.path.join(output_excel_path, excel_file_name)

        all_files_to_process_info: List[Dict[str, Any]] = [] # List of {'path', 'data', 'type', 'original_zip_name', 'folder_in_zip'}
        # This list will contain info for all individual image files from all zips

        try:
            # Phase 1: Extract all zips and collect all individual files to be processed
            for zip_content_bytes, zip_file_name_str in uploaded_files:
                try:
                    # Extract current zip to its own subdirectory in base_temp_dir
                    current_zip_extract_path = self.file_service.extract_zip_to_temp(
                        zip_content_bytes, base_temp_dir, zip_file_name_str
                    )
                    # Collect files from this specific extracted zip's directory structure
                    files_by_folder_in_zip: Dict[str, List[Dict[str, Any]]] = \
                        self.file_service.collect_files_from_extracted_zip(
                            current_zip_extract_path, zip_file_name_str
                        )

                    for folder_name_in_zip, files_info_list in files_by_folder_in_zip.items():
                        for file_info in files_info_list:
                            all_files_to_process_info.append({
                                **file_info, # 'path' (disk path), 'data', 'type', 'original_filename'
                                'original_zip_name': zip_file_name_str,
                                'folder_in_zip_key': folder_name_in_zip # Key for grouping in Excel later
                            })
                except Exception as e_zip:
                    logger.error(f"Failed to process zip {zip_file_name_str} for job {job_id}: {e_zip}", exc_info=True)
                    # Optionally, continue with other zips or fail the whole job
                    # For now, log and continue; individual file errors handled later.

            total_individual_files = len(all_files_to_process_info)
            _processed_jobs[job_id].total_files = total_individual_files
            logger.info(f"Job {job_id}: Collected {total_individual_files} total individual files for processing.")
            if total_individual_files == 0 and uploaded_files:
                 _processed_jobs[job_id].status = "failed"
                 _processed_jobs[job_id].error_message = "No processable files found in the uploaded ZIP(s)."
                 _processed_jobs[job_id].end_time = time.time()
                 logger.warning(f"Job {job_id} resulted in no processable files.")
                 # self.file_service.clean_temp_dir(base_temp_dir) # Clean up if job fails early
                 return # No Excel to generate

            # Phase 2: Process all collected files in batches
            aggregated_processing_results: Dict[str, List[MultimodalProcessingResult]] = {}
            # Key: folder_in_zip_key + original_zip_name (to make sheets unique if folders have same names across zips)
            # Value: List of MultimodalProcessingResult for files in that "folder"

            processed_file_count = 0
            batch_size = settings.BATCH_SIZE

            for i in range(0, total_individual_files, batch_size):
                current_batch_file_info = all_files_to_process_info[i : i + batch_size]
                logger.info(f"Job {job_id}: Processing batch {i//batch_size + 1}/{(total_individual_files + batch_size -1)//batch_size} "
                            f"with {len(current_batch_file_info)} files.")
                
                # Each item in current_batch_file_info has 'data', 'type', 'path' (full disk path for logging)
                # and also 'original_zip_name', 'folder_in_zip_key'
                batch_multimodal_results = self._process_document_batch_concurrently(current_batch_file_info)

                # Aggregate results by their original folder structure for Excel sheeting
                for idx, single_file_result in enumerate(batch_multimodal_results):
                    # Get the corresponding original file_info to find folder_key and zip_name
                    original_info_for_this_result = current_batch_file_info[idx] # Relies on order preservation
                    
                    # Construct a unique sheet key:
                    # Use original_info_for_this_result['folder_in_zip_key']
                    # and original_info_for_this_result['original_zip_name']
                    sheet_group_key = (f"{original_info_for_this_result['original_zip_name']}_"
                                       f"{original_info_for_this_result['folder_in_zip_key']}")

                    if sheet_group_key not in aggregated_processing_results:
                        aggregated_processing_results[sheet_group_key] = []
                    
                    # single_file_result.file_path should be the full disk path, useful for debugging
                    # For Excel, we might want the relative path within the zip or just filename
                    # The current Excel logic uses the disk path.
                    aggregated_processing_results[sheet_group_key].append(single_file_result)

                processed_file_count += len(current_batch_file_info)
                _processed_jobs[job_id].processed_files = processed_file_count
                if total_individual_files > 0:
                    _processed_jobs[job_id].progress_percentage = (processed_file_count / total_individual_files) * 100
                
                # Log progress
                time_elapsed = time.time() - job_start_time
                files_per_sec = processed_file_count / time_elapsed if time_elapsed > 0 else 0
                est_remaining_time = ((total_individual_files - processed_file_count) / files_per_sec) if files_per_sec > 0 else "N/A"
                _processed_jobs[job_id].estimated_time_remaining = est_remaining_time
                logger.info(
                    f"Job {job_id} Progress: {processed_file_count}/{total_individual_files} files "
                    f"({_processed_jobs[job_id].progress_percentage:.2f}%). "
                    f"Rate: {files_per_sec:.2f} files/sec. ETA: {est_remaining_time}s"
                )


            # Phase 3: Generate Excel report
            self._generate_excel_report(aggregated_processing_results, final_excel_file_path)

            _processed_jobs[job_id].status = "completed"
            _processed_jobs[job_id].output_file_path = final_excel_file_path
            logger.info(f"Job {job_id} completed. Excel report at: {final_excel_file_path}")

        except Exception as e_main:
            logger.error(f"Critical error during job {job_id} execution: {e_main}", exc_info=True)
            _processed_jobs[job_id].status = "failed"
            _processed_jobs[job_id].error_message = str(e_main)
            _processed_jobs[job_id].error_traceback = traceback.format_exc()
        finally:
            _processed_jobs[job_id].end_time = time.time()
            if _processed_jobs[job_id].total_files and _processed_jobs[job_id].processed_files == _processed_jobs[job_id].total_files :
                 _processed_jobs[job_id].progress_percentage = 100.0

            # Clean up the entire base temporary directory for the job *after* Excel is written (or if job fails badly)
            # Consider delaying cleanup if files need to be inspected on failure.
            # For now, cleaning up regardless of success/failure of this main processing logic.
            # self.file_service.clean_temp_dir(base_temp_dir) # Moved to task wrapper in endpoint

    def _generate_excel_report(self, folder_results_map: Dict[str, List[MultimodalProcessingResult]], excel_path: str):
        """
        Generates an Excel report from the processed document results.
        folder_results_map: Key is sheet_name_key, Value is list of MultimodalProcessingResult.
        """
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            for sheet_group_key, results_for_sheet in folder_results_map.items():
                if not results_for_sheet:
                    continue

                data_for_df = []
                for single_file_result in results_for_sheet:
                    # single_file_result.file_path is the full disk path of the image
                    # We might want to show a relative path or just original filename in Excel.
                    # For consistency with original, using file_path (which was disk path).
                    row_dict = {"filepath": single_file_result.file_path}
                    
                    if single_file_result.error: # If the whole file processing failed
                        row_dict["processing_error"] = single_file_result.error
                        # Add placeholders for fields if needed, or just error
                        for field_def in FIELDS + [{"name": "signature_coordinates"}]: # Ensure all potential cols are there
                            field_name = field_def["name"]
                            row_dict[field_name] = "ERROR"
                            row_dict[f"{field_name}_conf"] = 0.0
                            row_dict[f"{field_name}_reason"] = single_file_result.error
                        data_for_df.append(row_dict)
                        continue # Next file result

                    for extracted_field in single_file_result.extracted_fields:
                        field_name = extracted_field.field_name
                        row_dict[field_name] = extracted_field.value
                        row_dict[f"{field_name}_conf"] = extracted_field.confidence
                        if extracted_field.reason:
                            row_dict[f"{field_name}_reason"] = extracted_field.reason
                        # lang = extracted_field.language # if needed in excel
                    data_for_df.append(row_dict)

                if not data_for_df:
                    continue
                
                # Create DataFrame using Pydantic models for structure if desired, or directly
                # df = pd.DataFrame([ExcelRow(**row).model_dump() for row in data_for_df])
                df = pd.DataFrame(data_for_df)


                # Define column order (as per original logic)
                cols_ordered = ["filepath"]
                # Add field names, then their _conf, then _reason
                # Include "signature_coordinates" in this logic
                all_field_defs = FIELDS + [{"id": 99, "name": "signature_coordinates"}] # Add signature for column ordering

                for field_def in all_field_defs:
                    field_name_root = field_def["name"]
                    if field_name_root in df.columns:
                        cols_ordered.append(field_name_root)
                    if f"{field_name_root}_conf" in df.columns:
                        cols_ordered.append(f"{field_name_root}_conf")
                    if f"{field_name_root}_reason" in df.columns:
                        cols_ordered.append(f"{field_name_root}_reason")
                
                # Add any other columns that might have been added (e.g. processing_error)
                # and are not in the predefined order, to the end.
                remaining_cols = [col for col in df.columns if col not in cols_ordered]
                final_column_order = cols_ordered + remaining_cols
                
                # Filter df to only include existing columns in the specified order
                df = df.reindex(columns=[col for col in final_column_order if col in df.columns])

                # Sanitize sheet name
                # sheet_group_key might be like "zipfilename_foldername"
                clean_sheet_name = re.sub(r'[\\/*?[\]:]', '_', sheet_group_key)
                if len(clean_sheet_name) > EXCEL_SHEET_NAME_MAX_LEN:
                    clean_sheet_name = clean_sheet_name[:EXCEL_SHEET_NAME_MAX_LEN - 3] + "..."
                
                if not df.empty:
                    df.to_excel(writer, sheet_name=clean_sheet_name, index=False)
                    logger.info(f"Written sheet: {clean_sheet_name} to {excel_path}")
                else:
                    logger.warning(f"Skipped empty DataFrame for sheet: {clean_sheet_name}")
        logger.info(f"Excel report generated: {excel_path}")


    # --- Job Status Management ---
    def get_job_status(self, job_id: str) -> Optional[JobStatusDetail]:
        if job_id in _active_tasks:
            future = _active_tasks[job_id]
            job_detail = _processed_jobs.get(job_id) # Get latest from _processed_jobs

            if not job_detail: # Should ideally not happen if task is active
                return JobStatusDetail(job_id=job_id, status="unknown", error_message="Job detail missing while task active")

            if future.done():
                # Task finished, remove from active. Result should be in _processed_jobs.
                _active_tasks.pop(job_id, None)
                if future.exception():
                    logger.error(f"Job {job_id} (polled as active) actually failed with: {future.exception()}")
                    # Update _processed_jobs if it wasn't already updated by the wrapper
                    job_detail.status = "failed"
                    job_detail.error_message = str(future.exception())
                    job_detail.error_traceback = "".join(traceback.format_exception(future.exception()))
                    job_detail.end_time = time.time() # Ensure end time is set
                # If no exception, status in _processed_jobs should be 'completed' or 'failed' (by the wrapper)
                return _processed_jobs.get(job_id)
            else: # Still running
                # Return the current status from _processed_jobs which is updated periodically
                return job_detail
        
        return _processed_jobs.get(job_id)

    def add_active_task(self, job_id: str, future: concurrent.futures.Future):
        _active_tasks[job_id] = future

    def remove_active_task(self, job_id: str):
        return _active_tasks.pop(job_id, None)

    def update_job_details(self, job_id: str, updates: Dict[str, Any]):
        if job_id in _processed_jobs:
            # Convert JobStatusDetail to dict, update, then create new instance for validation
            current_job_dict = _processed_jobs[job_id].model_dump()
            current_job_dict.update(updates)
            try:
                _processed_jobs[job_id] = JobStatusDetail(**current_job_dict)
            except ValidationError as e:
                logger.error(f"Failed to update job {job_id} due to Pydantic validation error: {e}")
        else:
            logger.warning(f"Attempted to update non-existent job: {job_id}")


    def get_processed_job_detail(self, job_id: str) -> Optional[JobStatusDetail]:
        return _processed_jobs.get(job_id)