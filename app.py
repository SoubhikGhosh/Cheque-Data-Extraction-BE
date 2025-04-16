from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import logging
import io
import json
import psycopg2
from psycopg2.extras import Json, DictCursor
from typing import Dict, Any
from datetime import datetime
from pdf2image import convert_from_bytes
import uvicorn
from PIL import Image
import numpy as np
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cheque OCR API",
    description="API for extracting details from cheques using Google Gemini",
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

# Configure Google API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCD6DGeERwWQbBC6BK1Hq0ecagQj72rqyQ")
genai.configure(api_key=GOOGLE_API_KEY)

# Database configuration
DB_CONFIG = {
    'dbname': 'cheque_ocr',
    'user': os.getenv("DB_USER", "soubhikghosh"),
    'password': os.getenv("DB_PASSWORD", "99Ghosh"),
    'host': os.getenv("DB_HOST", "localhost"),
    'port': os.getenv("DB_PORT", "5432")
}

def get_db_connection():
    """Create and return a database connection."""
    try:
        # Connect to postgres to check if our database exists
        conn = psycopg2.connect(
            dbname='postgres',
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port']
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if our database exists
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_CONFIG['dbname'],))
        exists = cursor.fetchone()
        
        # Create database if it doesn't exist
        if not exists:
            logger.info(f"Database '{DB_CONFIG['dbname']}' does not exist. Creating...")
            cursor.execute(f"CREATE DATABASE {DB_CONFIG['dbname']}")
            
        cursor.close()
        conn.close()
        
        # Connect to our database
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

def init_db():
    """Initialize database tables."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create cheques table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cheques (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            processing_timestamp TIMESTAMP DEFAULT NOW(),
            extracted_data JSONB NOT NULL,
            file_data BYTEA,
            signature_coordinates JSONB
        )
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise

class ChequeProcessor:
    """Helper class for cheque processing using Gemini."""
    
    @staticmethod
    def process_cheque(file_data: bytes, file_type: str, file_name: str) -> Dict[str, Any]:
        """Process cheque images using Gemini's multimodal capabilities."""
        try:
            # First try gemini-1.5-pro for better quality, fallback to flash if needed
            try:
                model = genai.GenerativeModel("gemini-1.5-pro")
                logger.info(f"Using gemini-1.5-pro for cheque processing: {file_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize gemini-1.5-pro: {e}. Falling back to gemini-1.5-flash")
                model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Configure generation parameters for precise extraction
            generation_config = {
                "temperature": 0.01,      # Very low temperature for deterministic outputs
                "top_p": 0.99,            # High top_p for reliable responses
                "top_k": 40,              # Standard setting
                "max_output_tokens": 4096, # Ample tokens for detailed analysis
                "response_mime_type": "application/json"  # Request JSON response explicitly
            }
            
            # Apply the generation config
            model.generation_config = generation_config
            
            # Convert to image bytes if PDF
            image_bytes = file_data
            if file_type.lower() in ["application/pdf", "pdf"]:
                images = convert_from_bytes(file_data)
                if images:
                    # Just use the first page for cheques
                    img_byte_arr = BytesIO()
                    images[0].save(img_byte_arr, format="PNG")
                    image_bytes = img_byte_arr.getvalue()
                    file_type = "image/png"
            
            # General cheque extraction prompt
            extraction_prompt = """
            You are a banking document analysis expert specializing in Indian cheque processing.
            
            TASK: Analyze this cheque image and extract ALL key information with high precision.
            
            Extract the following fields:
            1. Bank Name and Details: The complete name of the bank and visible branch information
            2. Account Holder: The name of the account holder (if visible)
            3. Payee: The name of the entity to whom the cheque is payable
            4. Date: The complete date as written on the cheque
            5. Amount (Numerical): The amount in numbers including currency symbol
            6. Amount (Words): The complete amount written in words
            7. Cheque Number: The full cheque number
            8. Account Number: The complete account number
            9. Routing/IFSC Code: Any routing or IFSC code visible
            10. MICR Line: The complete MICR line at the bottom
            
            For each field, provide:
            - The exact value as it appears
            - A confidence score (0.0-1.0)
            - A brief reason for your confidence level
            
            Return ONLY a complete, properly formatted JSON object.
            """
            
            # Signature-specific detection prompt
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
            
            # Process in parallel for better performance
            extraction_future = None
            signature_future = None
            
            try:
                # Try to use asynchronous processing if supported
                extraction_future = model.generate_content_async([
                    extraction_prompt,
                    {"mime_type": file_type, "data": image_bytes}
                ])
                
                signature_future = model.generate_content_async([
                    signature_prompt,
                    {"mime_type": file_type, "data": image_bytes}
                ])
                
                # Wait for both to complete
                extraction_response = extraction_future.result()
                signature_response = signature_future.result()
            except AttributeError:
                # Fallback to sequential processing if async not available
                extraction_response = model.generate_content([
                    extraction_prompt,
                    {"mime_type": file_type, "data": image_bytes}
                ])
                
                signature_response = model.generate_content([
                    signature_prompt,
                    {"mime_type": file_type, "data": image_bytes}
                ])
            
            # Extract JSON from responses
            extracted_data = ChequeProcessor._extract_json(extraction_response.text)
            logger.info(extracted_data)
            signature_data = ChequeProcessor._extract_json(signature_response.text)
            
            logger.info(f"Signature detection results: {signature_data}")
            
            # Validate and fix signature data if needed
            if not signature_data or not signature_data.get('coordinates'):
                logger.warning(f"Invalid signature data returned by Gemini: {signature_data}")
                signature_data = {
                    "exists": True,
                    "coordinates": {
                        "x1": 0.65,
                        "y1": 0.55,
                        "x2": 0.98,
                        "y2": 0.85
                    },
                    "description": "Fallback signature area - Gemini extraction failed",
                    "confidence": 0.5
                }
            
            # Create comprehensive result
            result = {
                "file_name": file_name,
                "mime_type": file_type,
                "extracted_data": extracted_data,
                "signature_coordinates": signature_data,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during cheque processing: {str(e)}", exc_info=True)
            # Return error information but in a structured format for consistent handling
            return {
                "file_name": file_name,
                "mime_type": file_type,
                "error": str(e),
                "processing_timestamp": datetime.now().isoformat(),
                "signature_coordinates": {
                    "exists": False,
                    "coordinates": {
                        "x1": 0.65,
                        "y1": 0.55,
                        "x2": 0.98,
                        "y2": 0.85
                    },
                    "description": "Error fallback coordinates",
                    "confidence": 0.1
                },
                "extracted_data": {}
            }
       
    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Extract JSON from text that might contain markdown or extra content."""
        import json
        import re
        
        # Clean up potential JSON formatting
        if '```json' in text:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
            if match:
                text = match.group(1).strip()
        elif '```' in text:
            match = re.search(r'```\s*([\s\S]*?)\s*```', text)
            if match:
                text = match.group(1).strip()
        
        # Fix common issues: remove trailing commas which are invalid in JSON
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}, Text: {text[:100]}...")
            
            # Try even more aggressive JSON fixing
            try:
                # Use a third-party library if available
                try:
                    import json5
                    return json5.loads(text)
                except ImportError:
                    pass
                
                # Manual fallback approach
                # Remove all whitespace and try common fixes
                text = re.sub(r'\s+', '', text)
                text = text.replace(',}', '}').replace(',]', ']')
                # Fix missing quotes around keys
                text = re.sub(r'([{,])(\s*)([a-zA-Z0-9_]+)(\s*):', r'\1"\3":', text)
                return json.loads(text)
            except Exception:
                logger.error(f"Failed to fix JSON: {text[:200]}...")
                return {}
    
    @staticmethod
    def save_results_to_db(result: Dict[str, Any], file_data: bytes) -> int:
        """Save processing results to database."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Insert into cheques table
            cursor.execute('''
            INSERT INTO cheques (
                filename, 
                mime_type, 
                processing_timestamp,
                extracted_data,
                file_data,
                signature_coordinates
            ) VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            ''', (
                result.get("file_name", ""),
                result.get("mime_type", ""),
                datetime.now(),
                Json(result.get("extracted_data", {})),
                psycopg2.Binary(file_data),
                Json(result.get("signature_coordinates", {}))
            ))
            
            cheque_id = cursor.fetchone()[0]
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return cheque_id
        except Exception as e:
            logger.error(f"Error saving results to database: {str(e)}")
            raise

    @staticmethod
    def extract_signature(file_data: bytes, coords: Dict[str, Any]) -> bytes:
        """Extract signature from cheque image using coordinates with enhanced padding."""
        try:
            # Open image
            img = Image.open(BytesIO(file_data))
            width, height = img.size
            
            # Get coordinates (normalized from 0 to 1)
            # If coordinates are missing or invalid, use a default bottom right area
            coordinates = coords.get("coordinates", {})
            if not coordinates or not all(k in coordinates for k in ["x1", "y1", "x2", "y2"]):
                # Default to bottom right quadrant where signatures typically appear for Indian cheques
                coordinates = {
                    "x1": 0.65,
                    "y1": 0.60,
                    "x2": 0.98,
                    "y2": 0.90
                }
                logger.warning(f"Using default signature coordinates: {coordinates}")
            
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
            
            # Crop image
            signature_img = img.crop((x1, y1, x2, y2))
            
            # Optional: Enhance the signature image
            # Uncomment if needed:
            # from PIL import ImageOps
            # signature_img = signature_img.convert('L')  # Convert to grayscale
            # signature_img = ImageOps.autocontrast(signature_img, cutoff=0.5)  # Enhance contrast
            
            # Convert to bytes
            output = BytesIO()
            signature_img.save(output, format="PNG")
            
            return output.getvalue()
        except Exception as e:
            logger.error(f"Error extracting signature: {str(e)}")
            # If all else fails, try to return a portion of the bottom right of the image
            try:
                # Crop the bottom right quadrant as a fallback
                bottom_right = img.crop((int(width * 0.6), int(height * 0.6), width, height))
                output = BytesIO()
                bottom_right.save(output, format="PNG")
                return output.getvalue()
            except:
                raise

@app.post("/process-cheque/")
async def process_cheque(file: UploadFile = File(...)):
    """
    Upload and process a cheque image, returning extraction results as JSON.
    
    Supports PDF, JPG, PNG, and TIFF formats.
    """
    try:
        # Initialize database
        init_db()
        
        # Validate file type
        filename = file.filename
        file_extension = os.path.splitext(filename)[1].lower()
        
        valid_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif']
        if file_extension not in valid_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Please upload a PDF, JPG, PNG or TIFF file."
            )
            
        # Map extension to MIME type
        mime_type_map = {
            '.pdf': 'application/pdf',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff'
        }
        mime_type = mime_type_map[file_extension]
        
        # Read file content
        file_data = await file.read()
        
        # Process the cheque
        logger.info(f"Processing cheque: {filename}")
        result = ChequeProcessor.process_cheque(file_data, mime_type, filename)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Save results to database
        cheque_id = ChequeProcessor.save_results_to_db(result, file_data)
        logger.info(f"Saved cheque results to database with ID: {cheque_id}")
        
        # Add ID to result
        result["id"] = cheque_id
        
        # Remove file_data from result
        if "file_data" in result:
            del result["file_data"]
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing cheque: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/cheques/")
async def list_cheques(limit: int = 50, offset: int = 0):
    """
    List processed cheques with pagination.
    """
    init_db()
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM cheques")
        total = cursor.fetchone()[0]
        
        # Get cheques with corrected JSON path expressions
        cursor.execute('''
        SELECT 
            id, 
            filename, 
            mime_type,
            processing_timestamp,
            extracted_data->'Bank Name and Details'->>'value' as bank,
            extracted_data->'Payee'->>'value' as payee,
            extracted_data->'Amount (Numerical)'->>'value' as amount
        FROM cheques 
        ORDER BY processing_timestamp DESC 
        LIMIT %s OFFSET %s
        ''', (limit, offset))
        
        cheques = []
        for row in cursor.fetchall():
            cheque = dict(row)
            # Convert timestamps to strings
            cheque['processing_timestamp'] = cheque['processing_timestamp'].isoformat()
            cheques.append(cheque)
        
        cursor.close()
        conn.close()
        
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "cheques": cheques
        }
        
    except Exception as e:
        logger.error(f"Error listing cheques: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@app.get("/cheques/{cheque_id}")
async def get_cheque_details(cheque_id: int):
    """
    Get details for a specific cheque.
    """
    init_db()
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Get cheque details
        cursor.execute('''
        SELECT 
            id, 
            filename, 
            mime_type,
            processing_timestamp,
            extracted_data,
            signature_coordinates
        FROM cheques 
        WHERE id = %s
        ''', (cheque_id,))
        
        cheque = cursor.fetchone()
        
        if not cheque:
            raise HTTPException(status_code=404, detail=f"Cheque with ID {cheque_id} not found")
        
        # Convert to dict
        cheque_dict = dict(cheque)
        cheque_dict['processing_timestamp'] = cheque_dict['processing_timestamp'].isoformat()
        
        cursor.close()
        conn.close()
        
        return cheque_dict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cheque details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/cheques/{cheque_id}/signature")
async def get_signature_image(cheque_id: int):
    """
    Get the signature image cropped from a specific cheque.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Get cheque data and signature coordinates
        cursor.execute('''
        SELECT file_data, signature_coordinates, mime_type, filename
        FROM cheques
        WHERE id = %s
        ''', (cheque_id,))
        
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Cheque with ID {cheque_id} not found")
        
        file_data = result['file_data']
        coords = result['signature_coordinates']
        mime_type = result['mime_type']
        filename = result['filename']
        
        logger.info(f"Getting signature for cheque ID {cheque_id}, filename: {filename}")
        
        # Use the coordinates provided by Gemini
        if not coords or not coords.get('exists', False) or not coords.get('coordinates'):
            # If Gemini failed to return valid coordinates, use a generic fallback
            coords = {
                "exists": True,
                "coordinates": {
                    "x1": 0.65,
                    "y1": 0.55,
                    "x2": 0.98,
                    "y2": 0.85
                },
                "description": "Generic fallback signature area - Gemini extraction failed"
            }
            logger.warning(f"Using generic fallback signature area for cheque: {filename}")
        
        # Convert PDF to image if needed
        if mime_type.lower() in ["application/pdf", "pdf"]:
            images = convert_from_bytes(file_data)
            if images:
                img_byte_arr = BytesIO()
                images[0].save(img_byte_arr, format="PNG")
                file_data = img_byte_arr.getvalue()
        
        # Extract signature using coordinates from Gemini
        signature_image = ChequeProcessor.extract_signature(file_data, coords)
        
        cursor.close()
        conn.close()
        
        # Return signature image
        return StreamingResponse(
            BytesIO(signature_image),
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename=signature_{cheque_id}.png"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting signature: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/cheques/{cheque_id}/image")
async def get_cheque_image(cheque_id: int):
    """
    Get the original image for a specific cheque.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get cheque file data and mime type
        cursor.execute('''
        SELECT file_data, mime_type, filename
        FROM cheques
        WHERE id = %s
        ''', (cheque_id,))
        
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Cheque with ID {cheque_id} not found")
        
        file_data, mime_type, filename = result
        
        cursor.close()
        conn.close()
        
        # Sanitize filename to avoid encoding issues
        # Remove any non-ASCII characters and replace with underscore
        import re
        safe_filename = re.sub(r'[^\x00-\x7F]+', '_', filename)
        
        # Ensure filename is not empty after sanitization
        if not safe_filename or safe_filename.isspace():
            safe_filename = f"cheque_{cheque_id}.png"
        
        # Return file data with appropriate content type
        return StreamingResponse(
            BytesIO(file_data),
            media_type=mime_type,
            headers={"Content-Disposition": f"inline; filename={safe_filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cheque image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

if __name__ == "__main__":
    init_db()
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)