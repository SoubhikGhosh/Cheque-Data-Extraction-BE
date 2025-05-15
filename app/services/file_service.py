import os
import zipfile
import io
import tempfile
import shutil
import logging
from typing import List, Tuple, Dict, Optional
from app.constants import TEMP_DIR_PREFIX, SUPPORTED_MIME_TYPES

logger = logging.getLogger(__name__)

class FileService:
    def create_temp_dir(self, job_id: str) -> str:
        """Creates a unique temporary directory for a job."""
        temp_dir_path = tempfile.mkdtemp(prefix=f"{TEMP_DIR_PREFIX}{job_id}_")
        logger.info(f"Created temporary directory: {temp_dir_path} for job {job_id}")
        return temp_dir_path

    def clean_temp_dir(self, temp_dir_path: str):
        """Removes the temporary directory."""
        if os.path.exists(temp_dir_path):
            try:
                shutil.rmtree(temp_dir_path)
                logger.info(f"Cleaned up temporary directory: {temp_dir_path}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory {temp_dir_path}: {e}", exc_info=True)
        else:
            logger.warning(f"Attempted to clean non-existent temporary directory: {temp_dir_path}")


    def extract_zip_to_temp(self, zip_content: bytes, base_temp_dir: str, zip_file_name: str) -> str:
        """
        Extracts a single zip file into a subdirectory within the base temporary directory.
        Returns the path to the directory where files from this zip are extracted.
        """
        # Sanitize zip_file_name to create a valid directory name
        sanitized_zip_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in os.path.splitext(zip_file_name)[0])
        
        # Create a unique directory for this specific zip file's contents
        # to avoid conflicts if multiple zips have same internal folder structures.
        zip_specific_extract_path = os.path.join(base_temp_dir, f"zip_contents_{sanitized_zip_name}")
        os.makedirs(zip_specific_extract_path, exist_ok=True)
        
        try:
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                zf.extractall(zip_specific_extract_path)
            logger.info(f"Extracted {zip_file_name} to {zip_specific_extract_path}")
            return zip_specific_extract_path
        except zipfile.BadZipFile:
            logger.error(f"Bad zip file: {zip_file_name}")
            raise
        except Exception as e:
            logger.error(f"Failed to extract zip file {zip_file_name}: {e}", exc_info=True)
            raise

    def collect_files_from_extracted_zip(
        self,
        extracted_zip_path: str, # Path where one zip's contents were extracted
        original_zip_name: str # Name of the original zip file for context
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Walks through an extracted zip directory, collects processable files by their subfolder.
        Returns a dictionary where keys are 'folder_name' (relative path within the zip)
        and values are lists of file_info dictionaries {'path', 'data', 'type', 'original_filename'}.
        """
        folder_files_map: Dict[str, List[Dict[str, Any]]] = {}
        
        for root, _, files in os.walk(extracted_zip_path):
            if not files:  # Skip empty directories or directories with no files directly in them
                continue

            # Determine the 'folder_name' relative to the extracted_zip_path.
            # This represents the folder structure *within* the processed zip file.
            relative_folder_path = os.path.relpath(root, extracted_zip_path)
            
            # If files are at the root of the zip, relative_folder_path will be '.'
            # Use original_zip_name or a combination if multiple zips form the basis of folders
            # For this function, it's simpler to use the relative path or zip name if root.
            folder_key = relative_folder_path if relative_folder_path != '.' else f"{os.path.splitext(original_zip_name)[0]}_root"


            if folder_key not in folder_files_map:
                folder_files_map[folder_key] = []

            for file_name in files:
                if file_name.startswith('.') or file_name.startswith('~'): # Skip hidden/temp files
                    continue

                file_path = os.path.join(root, file_name)
                _, ext = os.path.splitext(file_name)
                file_ext_lower = ext.lower()

                if file_ext_lower not in SUPPORTED_MIME_TYPES:
                    logger.warning(f"Unsupported file type: {file_path} in zip {original_zip_name}")
                    continue
                
                mime_type = SUPPORTED_MIME_TYPES[file_ext_lower]
                
                try:
                    with open(file_path, 'rb') as f_in:
                        file_data = f_in.read()
                    
                    folder_files_map[folder_key].append({
                        'path': file_path, # Full path on disk for processing
                        'data': file_data,
                        'type': mime_type,
                        'original_filename': file_name # Filename within the zip/folder
                    })
                except Exception as e:
                    logger.error(f"Could not read file {file_path} from zip {original_zip_name}: {e}", exc_info=True)
        
        return folder_files_map