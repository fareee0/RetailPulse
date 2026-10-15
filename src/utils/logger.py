"""
RetailPulse Centralized Logger Utility
Provides structured logging to both console and persistent log files.
"""

import os
import sys
import logging
from typing import Optional

_LOGGER: Optional[logging.Logger] = None

def get_logger(name: str = "retailpulse", log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Initializes and returns a singleton-like configured logger.
    
    Args:
        name: Name of the logger module.
        log_dir: Directory where log files are stored.
        level: Logging verbosity level.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # Create logs directory if missing
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "retailpulse.log")

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Failed to create file handler for {log_file}: {e}")

    return logger
