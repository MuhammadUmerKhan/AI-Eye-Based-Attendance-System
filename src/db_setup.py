import warnings
from database import Database
from logger import get_logger

# Ignore warnings
warnings.filterwarnings("ignore")

# Configure logging
logger = get_logger(__name__)

def init_database():
    """
    Initialize SQLite database with students and attendance tables.
    """
    try:
        logger.debug("Starting database initialization")
        db = Database()
        db.init_tables()
        logger.info({"message": "Database initialized successfully"})
        return True, "Database initialized successfully"
    except Exception as e:
        logger.error({"error": str(e), "message": "Failed to initialize database"})
        return False, f"Failed to initialize database: {str(e)}"
    finally:
        logger.debug("Closing database connection in init_database")
        db.close_connection()

if __name__ == "__main__":
    success, message = init_database()
    print(message)