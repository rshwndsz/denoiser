import logging
import logging.handlers
import src.constants as C

logging.basicConfig(format=C.LOG_FORMAT)
logger  = logging.getLogger('memnet')
logger.setLevel(logging.DEBUG)
handler = logging.handlers.RotatingFileHandler(C.LOG_FILE, 
                                               maxBytes=(1048576*5),
                                               backupCount=7)
handler.setFormatter(logging.Formatter(C.LOG_FORMAT))
logger.addHandler(handler)


if __name__ == "__main__":
    logger.debug("Hello, MemNet 🐉")
