import logging
import os
from pathlib import Path

LOG_FILE_PATH = os.path.join("logs", "main.log")
Path(LOG_FILE_PATH).parent.mkdir(parents = True, exist_ok = True)
FILE_HANDLER = None
CONSOLE_HANDLER = None

def set_global_handlers(log_file_path:str):
    global FILE_HANDLER
    global CONSOLE_HANDLER

    
    file_handler = logging.FileHandler(log_file_path, mode = "w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    FILE_HANDLER = file_handler
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    CONSOLE_HANDLER = console_handler

def create_global_logger(name:str = "Main", level:int = logging.INFO) -> logging.Logger:
    global FILE_HANDLER
    global CONSOLE_HANDLER
    global LOG_FILE_PATH

    if FILE_HANDLER is None or CONSOLE_HANDLER is None:
        set_global_handlers(LOG_FILE_PATH)
    logger = logging.getLogger(name)  
    logger.setLevel(level) 
    logger.addHandler(FILE_HANDLER)
    logger.addHandler(CONSOLE_HANDLER)

    return logger



def create_logger(file_path:str, name:str = "Main", level:int = logging.INFO) -> logging.Logger:
    file_handler = logging.FileHandler(file_path, mode = "w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)

    return logger

