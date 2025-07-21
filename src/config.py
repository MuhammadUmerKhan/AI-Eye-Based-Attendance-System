import os

# Configuration settings for the attendance system MVP
DATABASE_PATH = os.path.join("database", "attendance.db")

def get_input_images_dir(course_id: str) -> str:
    """Return course-specific training images directory."""
    return os.path.join(INPUT_IMAGES_DIR, course_id)

def get_faiss_index_path(course_id: str) -> str:
    """Return course-specific FAISS index path."""
    return os.path.join("database", f"faiss_index_{course_id}.bin")

INPUT_IMAGES_DIR = os.path.join("images", "input_imgs")
TRAIN_IMAGES_DIT = os.path.join("images", "test_imgs")
LOG_FILE = os.path.join("logs", "attendance.log")
DEEPFACE_MODEL = "ArcFace"
FAISS_THRESHOLD = 0.4
EMBEDDING_DIM = 512

COURSES = {
    "AI": "Artificial Intelligence",
    "GD": "Graphic Design"
}