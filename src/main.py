from db_setup import init_database
from register_student import register_student
from mark_attendance import mark_attendance
from extract_embeddings import extract_embedding
from config import TRAIN_IMAGES_DIR
from database import Database
from logger import get_logger
import os, numpy as np

# Configure logging
logger = get_logger(__name__)

def main():
    try:
        # Initialize database
        db = Database()
        success, message = init_database()
        if not success:
            logger.error(message)
            print(message)
            return
        
        # Example: Register a student
        # image_path = os.path.join(TRAIN_IMAGES_DIR, "123.jpg")
        # success, message = register_student(db, "123", "John Doe", "Computer Science", image_path)
        # logger.info(message)
        # print(message)
        
        # embedding, _, error = extract_embedding("../images/train_imgs/123.jpg")
        # logger.info(f"Embeddings: {embedding}")
        # print(embedding)
        
        # Example: Mark attendance
        # student_id, name, message = mark_attendance(db, "../images/train_imgs/123.jpg")
        # logger.info(message)
        # print(message)
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        print(f"Error: {str(e)}")
    finally:
        db.close_connection()

def fetch_and_display_students():
    """
    Fetch and display all students from the database.
    """
    try:
        # Initialize database connection
        db = Database()
        logger.info({"message": "Fetching student data from database"})

        # Fetch students using the Database class
        students = db.fetch_students()
        if students is None:
            logger.warning({"message": "No students found or error occurred"})
            print("No students found or error occurred.")
            return

        # Display student data
        print("\n=== Registered Students ===")
        for student in students:
            student_id = student['id']
            name = student['name']
            # Convert embedding BLOB to numpy array for display (optional)
            embedding = np.frombuffer(student['embedding'], dtype=np.float32)
            embedding_summary = f"[length: {len(embedding)}]"  # Display embedding length instead of full array
            print(f"Embeddings: {embedding}")
            print(f"ID: {student_id}, Name: {name}, Embedding: {embedding_summary}")
            logger.info({"student_id": student_id, "name": name, "message": "Displayed student data"})

        print(f"Total students: {len(students)}")
        logger.info({"count": len(students), "message": "Completed fetching and displaying students"})

    except Exception as e:
        logger.error({"error": str(e), "message": "Failed to fetch student data"})
        print(f"Error: {str(e)}")

    finally:
        # Close database connection
        db.close_connection()

if __name__ == "__main__":
    fetch_and_display_students()