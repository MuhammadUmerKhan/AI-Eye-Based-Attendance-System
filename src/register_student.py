import numpy as np
from typing import Tuple
from extract_embeddings import extract_embedding
from logger import get_logger

logger = get_logger(__name__)

def register_student(db, student_id: str, name: str, department: str, image_input) -> Tuple[bool, str]:
    """
    Register a student by saving their ID, name, department, and eye region embedding to the database.
    Accepts a Database instance and either a file path or a file-like object for image_input.
    Returns success status and message.
    """
    try:
        embedding, _, error = extract_embedding(image_input)
        if embedding is None:
            logger.warning(f"Registration failed for student {student_id}: {error}")
            return False, error
        
        success, message = db.register_student(student_id, name, department, embedding)
        return success, message
    except Exception as e:
        logger.error(f"Error registering student {student_id}: {str(e)}")
        return False, f"Error: {str(e)}"