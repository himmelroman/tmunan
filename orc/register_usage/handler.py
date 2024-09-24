import logging
import os

from orc.sessions.db import DynamoDBSessionManager
from orc.sessions.models import UsageData

logger = logging.getLogger()
logger.setLevel("INFO")


def register_usage(event, context):

    # log
    logger.info(f"Received event: {event}")

    # init db
    session_manager = DynamoDBSessionManager(os.environ['DYNAMODB_TABLE'])

    # get event details
    user_id = event['detail']['user_id']
    session_id = event['detail']['session_id']
    usage_duration = event['detail']['usage_time_seconds']

    # update usage
    new_usage = UsageData(duration=int(usage_duration))
    success = session_manager.update_session_usage(user_id, session_id, new_usage)

    # log
    logger.info(f'Session usage update: {success=}')

    return True
