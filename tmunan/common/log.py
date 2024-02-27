import os
import logging
from pathlib import Path
from logging import DEBUG, INFO

# Add global log levels
STDOUT = INFO + 1
STDERR = INFO + 2
logging.addLevelName(STDOUT, 'STDOUT')
logging.addLevelName(STDERR, 'STDERR')
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# constants
LOG_DIR = Path(os.environ['WORK_DIR'] if 'WORK_DIR' in os.environ else '/tmp/') / 'log'
LOG_HEADER_START_TAG = '========================= LOG HEADER START ================================='
LOG_HEADER_END_TAG = '========================== LOG HEADER END =================================='


def get_log_header():

    header = f'\n{LOG_HEADER_START_TAG}\n' \
             f'commit={os.environ.get("GIT_COMMIT", "n/a")}, ' \
             f'branch={os.environ.get("GIT_BRANCH", "n/a")}, ' \
             f'tags={os.environ.get("GIT_TAGS", "n/a")} ' \
             f'\n{LOG_HEADER_END_TAG}\n'
    return header


def get_logger(log_name, log_path=None, console=True):

    # init log dir
    if not log_path:
        log_path = str(LOG_DIR / log_name)

    # get logger
    logger = logging.getLogger(log_name)

    # clear handlers (in case a logger with the same name existed)
    logger.handlers = []

    # clear root and parent handlers,
    # in case some other package (e.g. rollbar) caused a StreamHandler to be added by calling logging.basicConfig()
    logger.root.handlers = []
    logger.parent.handlers = []

    # set logger level
    logger.setLevel(DEBUG)

    # create console handler, if needed
    if console and not os.environ.get('PYTHON_LOG_DISABLE_CONSOLE', False):

        # create log handler
        ch = logging.StreamHandler()
        ch.setLevel(INFO)

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
        fh.setLevel(DEBUG)

        # add handler to logger
        logger.addHandler(fh)

    # save log path
    logger.log_path = log_path

    # add log header
    logger.debug(get_log_header())

    return logger
