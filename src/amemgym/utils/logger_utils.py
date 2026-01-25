import sys
from loguru import logger


def setup_logger(log_path):
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(log_path, level="TRACE", rotation="10 MB")
