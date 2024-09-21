import json
import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from orc.sessions.models import SessionItem, SessionInfo, UsageData


class DynamoDBSessionManager:

    def __init__(self, table_name: str):
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table(table_name)

        self.logger = logging.getLogger()
        self.logger.setLevel("INFO")

    def get_session(self, user_id: str, session_id: str) -> Optional[SessionItem]:
        try:
            response = self.table.get_item(
                Key={
                    'user_id': user_id,
                    'session_id': session_id
                }
            )
            if 'Item' in response:
                item = response['Item']
                return SessionItem(
                    user_id=item['user_id'],
                    session_id=item['session_id'],
                    info=SessionInfo(**json.loads(item['info'])),
                    usage=UsageData(**json.loads(item['usage']))
                )
            return None

        except ClientError:
            self.logger.exception("Error getting session")
            return None

    def create_session(self, session: SessionItem) -> bool:
        try:
            self.table.put_item(
                Item={
                    'user_id': session.user_id,
                    'session_id': session.session_id,
                    'info': json.dumps(session.info.model_dump()),
                    'usage': json.dumps(session.usage.model_dump())
                },
                ConditionExpression='attribute_not_exists(user_id) AND attribute_not_exists(session_id)'
            )
            return True

        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                self.logger.error(f"Session already exists for user {session.user_id} and session {session.session_id}")
            else:
                self.logger.exception(f"Error creating session")
            return False

    def update_session(self, session: SessionItem) -> bool:
        try:
            self.table.update_item(
                Key={
                    'user_id': session.user_id,
                    'session_id': session.session_id
                },
                UpdateExpression='SET info = :info, usage = :usage',
                ExpressionAttributeValues={
                    ':info': json.dumps(session.info.model_dump()),
                    ':usage': json.dumps(session.usage.model_dump())
                }
            )
            return True

        except ClientError:
            self.logger.exception("Error updating session")
            return False
