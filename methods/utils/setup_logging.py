import logging.config
import json
from pathlib import Path

def setup_logging():
    """
    Setup logging configuration from a JSON file.
    
    Args:
        None
    Returns:
        None
        
    Example usage:
    from setup_logging import setup_logging
    setup_logging()

    import logging
    logger = logging.getLogger(__name__)

    logger.debug("This is a debug message from the main script.")
    """
    path = Path(__file__).resolve().parent
    path = path / 'logging_config.json'
    with open(path, 'r') as f:
        config = json.load(f)
        logging.config.dictConfig(config)