from db_setup import init_database
from register_students import register_students
from mark_attendance import mark_attendance
from extract_embeddings import extract_embedding
from faiss_index import FaissIndex
from config import TRAIN_IMAGES_DIR
from database import Database
from logger import get_logger
import os, faiss
import numpy as np
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Configure logging
logger = get_logger(__name__)

def fetch_and_display_students():
    """
    Fetch and display all students from the database.
    """
    try:
        logger.debug("Initializing database connection for fetching students")
        db = Database()
        logger.info({"message": "Fetching student data from database"})

        logger.debug("Fetching students using Database class")
        students = db.fetch_students()
        if students is None:
            logger.warning({"message": "No students found or error occurred"})
            print("No students found or error occurred.")
            return

        print("\n=== Registered Students ===")
        for student in students:
            student_id = student['id']
            name = student['name']
            department = student['department']
            embedding = student['embedding']  # Already converted to numpy array in fetch_students
            embedding_summary = f"[{', '.join(map(str, embedding[:5]))}, ..., {', '.join(map(str, embedding[-5:]))}]"
            logger.debug(f"Displaying student {student_id}, embedding shape: {embedding.shape}, dtype: {embedding.dtype}, values: {embedding_summary}")
            print(f"ID: {student_id}, Name: {name}, Department: {department}, Embedding: [length: {len(embedding)}]")
            logger.info({"student_id": student_id, "name": name, "message": "Displayed student data", "embedding_shape": embedding.shape, "embedding_dtype": str(embedding.dtype), "embedding_summary": embedding_summary})

        print(f"Total students: {len(students)}")
        logger.info({"count": len(students), "message": "Completed fetching and displaying students"})

    except Exception as e:
        logger.error({"error": str(e), "message": "Failed to fetch student data"})
        print(f"Error: {str(e)}")

    finally:
        logger.debug("Closing database connection in fetch_and_display_students")
        db.close_connection()

def main():
    try:
        logger.debug("Starting main function")
        db = Database()
        logger.debug("Initializing database")
        success, message = init_database()
        if not success:
            logger.error({"message": message})
            print(message)
            return
        
        # Register a student to ensure 512-dimensional embeddings
        image_path = os.path.join(TRAIN_IMAGES_DIR, "123.jpg")
        # logger.debug(f"Registering student with image: {image_path}")
        # success, message = register_students(db, "123", "John Doe", "Computer Science", image_path)
        # if not success:
        #     logger.error({"message": message, "image_path": image_path})
        #     print(message)
        #     return
        
        # logger.info({"message": message})
        # print(message)
        
        # Initialize FAISS index after registration
        logger.debug("Initializing FAISS index")
        faiss_index = FaissIndex()
        
        # Fetch students to build FAISS index with 512-dimensional embeddings
        logger.debug("Fetching students for FAISS index")
        students = db.fetch_students()
        if students:
            logger.debug("Converting student embeddings to numpy array")
            embeddings = np.array([student['embedding'] for student in students], dtype=np.float32)
            for i, emb in enumerate(embeddings):
                if emb.shape[0] != 512:
                    logger.error({"student_id": students[i]['id'], "got": emb.shape[0], "expected": 512, "message": "Unexpected embedding dimension in database"})
                    print(f"Error: Unexpected embedding dimension {emb.shape[0]} for student {students[i]['id']}")
                    return
                embedding_summary = f"[{', '.join(map(str, emb[:5]))}, ..., {', '.join(map(str, emb[-5:]))}]"
                logger.debug(f"Student {students[i]['id']} database embedding values: {embedding_summary}")
            student_ids = [student['id'] for student in students]
            names = [student['name'] for student in students]
            logger.debug(f"Building FAISS index with {len(embeddings)} embeddings, shape: {embeddings.shape}, dtype: {embeddings.dtype}")
            faiss_index.build_index(embeddings, student_ids, names)
        else:
            logger.debug("No students found, setting FAISS index dimension to 512")
            faiss_index.dimension = 512
            faiss_index.index = faiss.IndexFlatL2(512)
            logger.info({"message": "Initialized empty FAISS index with dimension 512"})
        
        # Update FAISS index with new student embedding
        logger.debug("Updating FAISS index with new student")
        embedding, _, error = extract_embedding(image_path)
        if embedding is not None:
            logger.debug(f"New student embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
            embedding_summary = f"[{', '.join(map(str, embedding[:5]))}, ..., {', '.join(map(str, embedding[-5:]))}]"
            logger.debug(f"DeepFace embedding values: {embedding_summary}")
            if embedding.shape[0] != 512:
                logger.error({"got": embedding.shape[0], "expected": 512, "message": "Unexpected embedding dimension from extract_embedding"})
                print(f"Error: Unexpected embedding dimension {embedding.shape[0]}")
                return
            embedding = embedding.astype(np.float32)
            faiss_index.update_index(embedding, "123", "John Doe")
        else:
            logger.error({"error": error, "message": "Failed to extract embedding for FAISS update"})
            print(f"Error: {error}")
            return
        
        # Mark attendance
        logger.debug(f"Marking attendance with image: {image_path}")
        student_id, name, message = mark_attendance(db, image_path, faiss_index)
        logger.info({"message": message})
        print(message)
        
        # Fetch and display students
        # logger.debug("Fetching and displaying students")
        # fetch_and_display_students()
        
    except Exception as e:
        logger.error({"error": str(e), "message": "Unexpected error in main"})
        print(f"Error: {str(e)}")
    finally:
        logger.debug("Closing database connection in main")
        db.close_connection()

if __name__ == "__main__":
    main()