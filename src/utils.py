import os
import cv2
import numpy as np
from logger import get_logger

# Configure logging
logger = get_logger(__name__)

def save_image(file, output_path):
    """
    Save an uploaded image to the specified path.
    Returns success status and message.
    """
    try:
        # Read the uploaded file
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save image
        cv2.imwrite(output_path, img)
        logger.info(f"Image saved to {output_path}")
        return True, f"Image saved to {output_path}"
    except Exception as e:
        logger.error(f"Failed to save image to {output_path}: {str(e)}")
        return False, f"Failed to save image: {str(e)}"

def cleanup_temp_image(image_path):
    """
    Delete a temporary image file.
    """
    try:
        if os.path.exists(image_path):
            os.remove(image_path)
            logger.info(f"Temporary image deleted: {image_path}")
    except Exception as e:
        logger.error(f"Failed to delete temporary image {image_path}: {str(e)}")