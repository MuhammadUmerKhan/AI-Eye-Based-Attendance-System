import numpy as np
import warnings
from typing import Tuple, Optional
from datetime import datetime
from extract_embeddings import extract_embedding
from faiss_index import FaissIndex
from config import FAISS_THRESHOLD
from logger import get_logger

# Ignore warnings
warnings.filterwarnings("ignore")

# Configure logging
logger = get_logger(__name__)

def mark_attendance(db, image_input, faiss_index: FaissIndex) -> Tuple[Optional[str], Optional[str], str]:
    """
    Match an image's embedding against stored embeddings and mark attendance.
    Accepts a Database instance, image input (file path or file-like object), and a FaissIndex instance.
    Returns student ID, name, and message.
    """
    try:
        logger.debug(f"Starting attendance marking process for image: {image_input}")
        logger.debug(f"Extracting embedding from image input: {image_input}")
        embedding, _, error = extract_embedding(image_input)
        if embedding is None:
            logger.warning({"error": error, "message": "Attendance marking failed"})
            return None, None, error
        
        logger.debug(f"Extracted embedding shape: {embedding.shape}")
        logger.debug("Fetching students from database")
        students = db.fetch_students()
        if not students:
            logger.warning({"message": "No students registered in database"})
            return None, None, "No students registered"
        
        logger.info({"count": len(students), "message": "Fetched student data for attendance", "query_embedding_shape": embedding.shape})
        student_ids = [student['id'] for student in students]
        names = [student['name'] for student in students]
        logger.debug(f"Retrieved {len(student_ids)} student IDs and names")
        
        # Search for nearest neighbor using the provided FAISS index
        logger.debug(f"Searching FAISS index with embedding shape: {embedding.shape}")
        D, I = faiss_index.search(np.array([embedding], dtype=np.float32))
        if D is None or I is None:
            logger.error({"message": "FAISS search returned None, possibly due to uninitialized index or dimension mismatch"})
            return None, None, "FAISS search failed: Index not initialized or dimension mismatch"
        
        distance = D[0][0]
        logger.debug(f"FAISS search result: distance={distance:.2f}, index={I[0][0]}")
        
        if distance < FAISS_THRESHOLD:
            if I[0][0] >= len(student_ids):
                logger.error({"index": I[0][0], "student_ids_length": len(student_ids), "message": "Invalid index returned by FAISS"})
                return None, None, "FAISS search returned invalid index"
            student_id = student_ids[I[0][0]]
            name = names[I[0][0]]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.debug(f"Marking attendance for student_id: {student_id}, name: {name}, timestamp: {timestamp}")
            success, message = db.mark_attendance(student_id, timestamp)
            if success:
                logger.info({"student_id": student_id, "name": name, "distance": distance, "message": f"Attendance marked for {name}"})
                return student_id, name, f"Attendance marked for {name} (ID: {student_id})"
            else:
                logger.error({"student_id": student_id, "error": message, "message": "Attendance marking failed"})
                return None, None, message
        else:
            logger.warning({"distance": distance, "message": f"No match found (distance: {distance:.2f})"})
            return None, None, f"No match found (distance: {distance:.2f})"
            
    except Exception as e:
        logger.error({"error": str(e), "message": f"Error during attendance marking for {image_input}"})
        return None, None, f"Error during attendance marking: {str(e)}"