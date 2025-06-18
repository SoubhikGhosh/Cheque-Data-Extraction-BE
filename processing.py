# /cheque_extraction_api/processing.py

import os
import io
import zipfile
import tempfile
import time
import logging
import json
import traceback
from typing import List, Dict, Any
import concurrent.futures

import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, Part

import config
import prompts
import utils

logger = logging.getLogger(__name__)

def process_single_document(file_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes a single cheque image using Vertex AI to extract fields.

    Args:
        file_info: A dictionary containing file path, data, and type.

    Returns:
        A dictionary with the extracted results or an error.
    """
    file_path = file_info['path']
    file_data = file_info['data']
    file_type = file_info['type']
    
    logger.info(f"Starting processing for: {os.path.basename(file_path)}")

    try:
        model = GenerativeModel(config.MODEL_NAME, safety_settings=config.SAFETY_SETTINGS)
        file_part = Part.from_data(data=file_data, mime_type=file_type)
        extraction_prompt = prompts.get_extraction_prompt()

        response = utils.call_vertex_ai_with_retry(model, [extraction_prompt, file_part])
        
        json_text = utils.extract_json_from_text(response.text)
        
        try:
            result = json.loads(json_text)
            result['file_path'] = file_path
            logger.info(f"Successfully processed and parsed JSON for: {os.path.basename(file_path)}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed for {os.path.basename(file_path)}. Error: {e}. Response snippet: {json_text[:500]}...")
            return {"error": f"JSON Decode Error: {e}", "file_path": file_path, "extracted_fields": []}

    except Exception as e:
        logger.error(f"An unexpected error occurred while processing {os.path.basename(file_path)}: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e), "file_path": file_path, "extracted_fields": []}


def process_zip_file_and_generate_report(job_id: str, file_contents: List[bytes], file_names: List[str], job_status_dict: Dict):
    """
    Orchestrates the entire process of unzipping files, processing them in batches,
    and generating a final Excel report. This function is designed to be run in a
    background thread.
    """
    job_start_time = time.time()
    
    try:
        with tempfile.TemporaryDirectory(prefix=f"job_{job_id}_") as temp_dir:
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            all_files_to_process = []
            folder_map = {} # Maps folder name to a list of file paths

            # Step 1: Unzip all files and collect files to process
            for zip_content, zip_name in zip(file_contents, file_names):
                with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                    zf.extractall(temp_dir)
                    for info in zf.infolist():
                        if not info.is_dir() and not info.filename.startswith('__MACOSX'):
                            full_path = os.path.join(temp_dir, info.filename)
                            folder_name = os.path.basename(os.path.dirname(full_path))
                            if folder_name not in folder_map:
                                folder_map[folder_name] = []
                            folder_map[folder_name].append(full_path)
            
            # Step 2: Read file data and prepare for batching
            for full_path in [p for paths in folder_map.values() for p in paths]:
                ext = os.path.splitext(full_path)[1].lower()
                mime_types = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.tiff': 'image/tiff', '.tif': 'image/tiff'}
                if ext in mime_types:
                    with open(full_path, 'rb') as f:
                        all_files_to_process.append({'path': full_path, 'data': f.read(), 'type': mime_types[ext]})
                else:
                    logger.warning(f"Skipping unsupported file type: {full_path}")

            total_files = len(all_files_to_process)
            job_status_dict["total_files"] = total_files
            logger.info(f"Job {job_id}: Found {total_files} processable images.")

            # Step 3: Process all files in parallel batches
            all_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS, thread_name_prefix=f'Processor_{job_id}') as executor:
                futures = [executor.submit(process_single_document, file_info) for file_info in all_files_to_process]
                for future in concurrent.futures.as_completed(futures):
                    all_results.append(future.result())
                    processed_count = len(all_results)
                    job_status_dict["processed_files"] = processed_count
                    if total_files > 0:
                       job_status_dict["progress_percentage"] = (processed_count / total_files) * 100
                    logger.info(f"Job {job_id}: Progress {processed_count}/{total_files} ({job_status_dict['progress_percentage']:.2f}%)")

            # Step 4: Generate Excel Report
            excel_path = os.path.join(output_dir, f"cheque_extraction_results_{job_id}.xlsx")
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                for folder_name, file_paths in folder_map.items():
                    # Filter results for the current folder
                    folder_results = [res for res in all_results if os.path.basename(os.path.dirname(res['file_path'])) == folder_name]
                    if not folder_results:
                        continue
                        
                    data_for_df = []
                    for result in folder_results:
                        row = {'filepath': os.path.basename(result['file_path'])}
                        for field in result.get('extracted_fields', []):
                            fname = field.get('field_name')
                            if fname:
                                row[fname] = field.get('value')
                                row[f'{fname}_conf'] = field.get('confidence')
                                if field.get('reason'):
                                    row[f'{fname}_reason'] = field.get('reason')
                        data_for_df.append(row)

                    df = pd.DataFrame(data_for_df)
                    
                    # Order columns
                    base_cols = ['filepath']
                    field_cols = []
                    for field in config.FIELDS:
                        fname = field["name"]
                        if fname in df.columns:
                           field_cols.append(fname)
                           field_cols.append(f'{fname}_conf')
                           if f'{fname}_reason' in df.columns:
                               field_cols.append(f'{fname}_reason')
                    
                    df = df[base_cols + field_cols]
                    
                    sheet_name = re.sub(r'[\\/*?[\]:]', '_', folder_name)
                    sheet_name = (sheet_name[:28] + '...') if len(sheet_name) > 31 else sheet_name
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            job_status_dict.update({
                "status": "completed",
                "output_file_path": excel_path,
                "end_time": time.time(),
                "processing_time": time.time() - job_start_time,
            })
            logger.info(f"Job {job_id} completed successfully. Report at {excel_path}")

    except Exception as e:
        logger.error(f"Critical error in job {job_id}: {e}")
        logger.error(traceback.format_exc())
        job_status_dict.update({
            "status": "failed",
            "error_message": str(e),
            "end_time": time.time()
        })