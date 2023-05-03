import logging
import sys

def setup_logger(name=__name__):
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    c_handler = logging.StreamHandler(sys.stdout)
    
    c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)

    logger.addHandler(c_handler)

    return logger
