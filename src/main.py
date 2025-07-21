from src.db_setup import init_database
from src.register_students import register_student
from src.mark_attendance import mark_attendance
from src.extract_embeddings import extract_embedding
from src.faiss_index import FaissIndex
from src.config import get_train_images_dir, EMBEDDING_DIM, COURSES
from src.database import Database
from src.logger import get_logger
import numpy as np
import warnings
from typing import Optional

# Ignore warnings
warnings.filterwarnings("ignore")

# Configure logging
logger = get_logger(__name__)

def fetch_and_display_students(db: Database, course_id: Optional[str] = None):
    """
    Fetch and display all students from the database, optionally filtered by course.
    """
    try:
        logger.debug(f"Fetching students from database, course_id: {course_id if course_id else 'all'}")
        students = db.fetch_students(course_id)
        if students is None:
            logger.warning({"message": "No students found or error occurred"})
            print("No students found or error occurred.")
            return

        print(f"\n=== Registered Students (Course: {course_id if course_id else 'All'}) ===")
        for student in students:
            student_id = student['id']
            name = student['name']
            course_name = student['course_name']
            embedding = student['embedding']
            print(f"ID: {student_id}, Name: {name}, Course: {course_name}, Embedding: [length: {len(embedding)}]")
            logger.info({"student_id": student_id, "course_id": course_id, "name": name, "message": "Displayed student data"})
        print(f"Total students: {len(students)}")
        logger.info({"count": len(students), "course_id": course_id if course_id else "all", "message": "Completed fetching and displaying students"})
    except Exception as e:
        logger.error({"error": str(e), "message": "Failed to fetch student data"})
        print(f"Error: {str(e)}")

def main():
    try:
        logger.debug("Starting main function")
        with Database() as db:
            logger.debug("Initializing database")
            success, message = init_database()
            if not success:
                logger.error({"message": message})
                print(message)
                return
            
            course_id = "AI"
            course_name = COURSES[course_id]
            image_path = os.path.join(get_train_images_dir(course_id), "1234.jpg")
            faiss_index = FaissIndex()
            
            logger.debug(f"Fetching students for course {course_id}")
            students = db.fetch_students(course_id)
            if students:
                logger.debug(f"Converting student embeddings to numpy array for course {course_id}")
                embeddings = np.array([student['embedding'] for student in students], dtype=np.float32)
                for i, emb in enumerate(embeddings):
                    if emb.shape[0] != EMBEDDING_DIM:
                        logger.error({"student_id": students[i]['id'], "course_id": course_id, "got": emb.shape[0], "expected": EMBEDDING_DIM, "message": "Unexpected embedding dimension in database"})
                        print(f"Error: Unexpected embedding dimension {emb.shape[0]} for student {students[i]['id']}")
                        return
                student_ids = [student['id'] for student in students]
                names = [student['name'] for student in students]
                logger.debug(f"Building FAISS index for course {course_id} with {len(embeddings)} embeddings")
                faiss_index.build_index(embeddings, student_ids, names, course_id)
            else:
                logger.debug(f"No students found for course {course_id}, loading empty FAISS index")
                faiss_index.load_index(course_id)
            
            logger.debug(f"Updating FAISS index with new student for course {course_id}")
            embedding, _, error = extract_embedding(image_path)
            if embedding is not None:
                if embedding.shape[0] != EMBEDDING_DIM:
                    logger.error({"got": embedding.shape[0], "expected": EMBEDDING_DIM, "message": "Unexpected embedding dimension from extract_embedding"})
                    print(f"Error: Unexpected embedding dimension {embedding.shape[0]}")
                    return
                embedding = embedding.astype(np.float32)
                success, message = register_student(db, 1234, "John Doe", course_id, course_name, image_path, faiss_index)
                if not success:
                    logger.error({"error": message, "message": "Failed to register student"})
                    print(f"Error: {message}")
                    return
            else:
                logger.error({"error": error, "message": "Failed to extract embedding for FAISS update"})
                print(f"Error: {error}")
                return
            
            logger.debug(f"Marking attendance with image: {image_path} for course {course_id}")
            student_id, name, message = mark_attendance(db, image_path, faiss_index, course_id)
            logger.info({"message": message})
            print(message)
    except Exception as e:
        logger.error({"error": str(e), "message": "Unexpected error in main"})
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()