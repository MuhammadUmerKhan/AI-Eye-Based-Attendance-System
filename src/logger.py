import logging

# Configure logging
logging.basicConfig(
    filename="../logs/attendance.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def get_logger(name):
    """Return a logger instance."""
    return logging.getLogger(name)