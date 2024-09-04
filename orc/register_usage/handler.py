import logging

logger = logging.getLogger()
logger.setLevel("INFO")


def register_usage(event, context):

    # print incoming event
    logger.info(f'RegisterUsage Invoked: {event=}')

    return True
