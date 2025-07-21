import numpy as np
import warnings
from typing import Tuple, Optional
from datetime import datetime
from src.extract_embeddings import extract_embedding
from src.faiss_index import FaissIndex
from src.config import FAISS_THRESHOLD, EMBEDDING_DIM, COURSES
from src.logger import get_logger

# Ignore warnings
warnings.filterwarnings("ignore")

# Configure logging
logger = get_logger(__name__)

def mark_attendance(db, image_input, faiss_index: FaissIndex, course_id: str) -> Tuple[Optional[int], Optional[str], str]:
    """
    Match an image's embedding against stored embeddings for a specific course and mark attendance.
    Accepts a Database instance, image input (file path or file-like object), a FaissIndex instance, and course_id.
    Returns student ID, name, and message.
    """
    try:
        logger.debug(f"Starting attendance marking for image in course {course_id}")
        embedding, _, error = extract_embedding(image_input)
        if embedding is None:
            logger.warning({"error": error, "message": "Attendance marking failed"})
            return None, None, error
        
        students = db.fetch_students(course_id)
        if not students:
            logger.warning({"course_id": course_id, "message": "No students registered in course"})
            return None, None, f"Sorry, you are not registered in this course"
        
        logger.info({"count": len(students), "message": f"Fetched student data for attendance in course {course_id}"})
        
        D, I, student_ids, names = faiss_index.search(np.array([embedding], dtype=np.float32), course_id)
        if D is None or I is None:
            logger.error({"course_id": course_id, "message": "FAISS search failed due to uninitialized index or dimension mismatch"})
            return None, None, f"Sorry, you are not registered in this course"
        
        distance = D[0][0]
        logger.debug(f"FAISS search result: distance={distance:.2f}, index={I[0][0]}")
        
        if distance < FAISS_THRESHOLD:
            if I[0][0] >= len(student_ids):
                logger.error({"index": I[0][0], "student_ids_length": len(student_ids), "message": "Invalid index returned by FAISS"})
                return None, None, "FAISS search returned invalid index"
            student_id = student_ids[I[0][0]]
            name = names[I[0][0]]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            success, message = db.mark_attendance(student_id, timestamp, course_id)
            if success:
                logger.info({"student_id": student_id, "name": name, "course_id": course_id, "distance": distance, "message": f"Attendance marked for {name}"})
                return student_id, name, f"Attendance marked for {name} (ID: {student_id}) in course {course_id}"
            logger.error({"student_id": student_id, "course_id": course_id, "error": message, "message": "Attendance marking failed"})
            return None, None, message
        logger.warning({"distance": distance, "course_id": course_id, "message": f"No match found (distance: {distance:.2f})"})
        return None, None, f"Sorry, you are not registered in this course"
    except Exception as e:
        logger.error({"error": str(e), "message": f"Error during attendance marking in course {course_id}"})
        return None, None, f"Error during attendance marking: {str(e)}"