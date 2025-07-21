import logging
import warnings
import os
from src.config import LOG_FILE

# Ignore warnings
warnings.filterwarnings("ignore")

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def get_logger(name):
    """Return a logger instance."""
    return logging.getLogger(name)