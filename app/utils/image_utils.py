import io
import logging
from PIL import Image
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

def crop_image_with_coordinates(image_bytes: bytes, coordinates: Dict[str, float]) -> Optional[bytes]:
    """
    Crop an image based on normalized coordinates with extra padding.
    
    Args:
        image_bytes (bytes): Original image as bytes
        coordinates (dict): Normalized coordinates (x1, y1, x2, y2)
    
    Returns:
        bytes: Cropped image as bytes in PNG format, or None on error.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        
        # Default coordinates from original if not fully provided (though schema should enforce)
        x1_norm = coordinates.get("x1", 0.65)
        y1_norm = coordinates.get("y1", 0.60) # Original was 0.55, prompt says 0.55 but example 0.60? Using 0.60
        x2_norm = coordinates.get("x2", 0.98)
        y2_norm = coordinates.get("y2", 0.90) # Original was 0.85, prompt says 0.85 but example 0.90? Using 0.90

        x1 = int(x1_norm * width)
        y1 = int(y1_norm * height)
        x2 = int(x2_norm * width)
        y2 = int(y2_norm * height)
        
        # Ensure x1 < x2 and y1 < y2
        if x1 >= x2: x1 = x2 -1 # Ensure at least 1 pixel width if coords are bad
        if y1 >= y2: y1 = y2 -1 # Ensure at least 1 pixel height

        padding_x_abs = int((x2 - x1) * 0.30)
        padding_y_abs = int((y2 - y1) * 0.30)
        
        # Apply padding while ensuring we stay within image bounds
        final_x1 = max(0, x1 - padding_x_abs)
        final_y1 = max(0, y1 - padding_y_abs)
        final_x2 = min(width, x2 + padding_x_abs)
        final_y2 = min(height, y2 + padding_y_abs)
        
        logger.info(f"Cropping: Original dims ({width}x{height}). Coords: Norm ({x1_norm},{y1_norm})-({x2_norm},{y2_norm}). "
                    f"Pixel ({x1},{y1})-({x2},{y2}). Padded ({final_x1},{final_y1})-({final_x2},{final_y2})")

        if final_x1 >= final_x2 or final_y1 >= final_y2:
            logger.error(f"Invalid crop dimensions after padding: ({final_x1},{final_y1})-({final_x2},{final_y2}). Skipping crop.")
            return None

        cropped_img = img.crop((final_x1, final_y1, final_x2, final_y2))
        
        img_byte_arr = io.BytesIO()
        cropped_img.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

    except Exception as e:
        logger.error(f"Error cropping image: {e}", exc_info=True)
        return None