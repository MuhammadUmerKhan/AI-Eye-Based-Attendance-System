from src.db_setup import init_database
from src.register_students import register_students
from src.mark_attendance import mark_attendance
from src.extract_embeddings import extract_embedding
from src.faiss_index import FaissIndex
from src.config import TRAIN_IMAGES_DIR, EMBEDDING_DIM
from src.database import Database
from src.logger import get_logger
import numpy as np
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Configure logging
logger = get_logger(__name__)

def fetch_and_display_students(db: Database):
    """
    Fetch and display all students from the database.
    """
    try:
        logger.debug("Fetching students from database")
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
            embedding = student['embedding']
            print(f"ID: {student_id}, Name: {name}, Department: {department}, Embedding: [length: {len(embedding)}]")
            logger.info({"student_id": student_id, "name": name, "message": "Displayed student data"})
        print(f"Total students: {len(students)}")
        logger.info({"count": len(students), "message": "Completed fetching and displaying students"})
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
            
            image_path = os.path.join(TRAIN_IMAGES_DIR, "123.jpg")
            faiss_index = FaissIndex()
            
            logger.debug("Fetching students for FAISS index")
            students = db.fetch_students()
            if students:
                logger.debug("Converting student embeddings to numpy array")
                embeddings = np.array([student['embedding'] for student in students], dtype=np.float32)
                for i, emb in enumerate(embeddings):
                    if emb.shape[0] != EMBEDDING_DIM:
                        logger.error({"student_id": students[i]['id'], "got": emb.shape[0], "expected": EMBEDDING_DIM, "message": "Unexpected embedding dimension in database"})
                        print(f"Error: Unexpected embedding dimension {emb.shape[0]} for student {students[i]['id']}")
                        return
                student_ids = [student['id'] for student in students]
                names = [student['name'] for student in students]
                logger.debug(f"Building FAISS index with {len(embeddings)} embeddings")
                faiss_index.build_index(embeddings, student_ids, names)
            else:
                logger.debug("No students found, setting FAISS index dimension to 512")
                faiss_index.dimension = EMBEDDING_DIM
                faiss_index.index = faiss.IndexFlatL2(EMBEDDING_DIM)
                logger.info({"message": "Initialized empty FAISS index with dimension 512"})
            
            logger.debug("Updating FAISS index with new student")
            embedding, _, error = extract_embedding(image_path)
            if embedding is not None:
                if embedding.shape[0] != EMBEDDING_DIM:
                    logger.error({"got": embedding.shape[0], "expected": EMBEDDING_DIM, "message": "Unexpected embedding dimension from extract_embedding"})
                    print(f"Error: Unexpected embedding dimension {embedding.shape[0]}")
                    return
                embedding = embedding.astype(np.float32)
                faiss_index.update_index(embedding, "123", "John Doe")
            else:
                logger.error({"error": error, "message": "Failed to extract embedding for FAISS update"})
                print(f"Error: {error}")
                return
            
            logger.debug(f"Marking attendance with image: {image_path}")
            student_id, name, message = mark_attendance(db, image_path, faiss_index)
            logger.info({"message": message})
            print(message)
    except Exception as e:
        logger.error({"error": str(e), "message": "Unexpected error in main"})
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()