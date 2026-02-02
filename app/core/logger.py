import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance.
    Standardizes log format across the application for observability.
    """
    logger = logging.getLogger(name)
    
    # Singleton-like setup: Only add handler if not already present
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler(sys.stdout)
        
        # Format: Timestamp | Level | Module:Function:Line | Message
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Prevent propagation to root logger to duplicate logs if root is configured elsewhere
        logger.propagate = False
        
    return logger
