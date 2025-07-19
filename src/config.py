import os

# Configuration settings for the attendance system MVP
DATABASE_PATH = os.path.join("..", "database", "attendance.db")
TRAIN_IMAGES_DIR = os.path.join("..", "images", "train_imgs")
INPUT_IMAGES_DIR = os.path.join("..", "images", "input_imgs")
LOG_FILE = os.path.join("..", "logs", "attendance.log")
DEEPFACE_MODEL = "ArcFace"  # DeepFace model for embedding extraction
FAISS_THRESHOLD = 0.4       # L2 distance threshold for matchingQ