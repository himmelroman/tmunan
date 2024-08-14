import os
import logging
from pathlib import Path
from logging import INFO

# Add global log levels
STDOUT = INFO + 1
STDERR = INFO + 2
logging.addLevelName(STDOUT, 'STDOUT')
logging.addLevelName(STDERR, 'STDERR')
# logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.basicConfig(format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',)

# constants
LOG_DIR = Path(os.environ['WORK_DIR'] if 'WORK_DIR' in os.environ else '/tmp/') / 'log'


def get_logger(log_name, log_path=None, console=True):

    # read log level
    log_level = os.environ.get('LOG_LEVEL', 'INFO')
    log_level = logging.INFO  # logging.getLevelNamesMapping()[log_level]

    # init log dir
    if not log_path:
        log_path = str(LOG_DIR / log_name)

    # get logger
    logger = logging.getLogger(log_name)

    # set logger level
    logger.setLevel(log_level)

    # create console handler, if needed
    if console and not os.environ.get('PYTHON_LOG_DISABLE_CONSOLE', False):

        # create log handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)

        # Add console handler to root logger
        # Other packages might register a StreamHandler when calling logging.basicConfig() (e.g. rollbar)
        logging.basicConfig(handlers=[ch])

    # create file handler, if needed
    if log_path:

        # create dir
        log_path = Path(log_path)
        Path(log_path.parent).mkdir(parents=True, exist_ok=True)

        # init file handle
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setLevel(log_level)

        # add handler to logger
        logger.addHandler(fh)

    # save log path
    logger.log_path = log_path

    return logger
