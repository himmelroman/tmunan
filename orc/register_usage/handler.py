import logging

logger = logging.getLogger()
logger.setLevel("INFO")


def run_session(event, context):

    # print incoming event
    logger.info(f'RegisterUsage Invoked: {event=}')

    return True
