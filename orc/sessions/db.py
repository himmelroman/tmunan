import json
import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from orc.sessions.models import SessionItem, SessionData, UsageData


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
                    session_data=SessionData(**json.loads(item['session_data'])),
                    usage_data=UsageData(**json.loads(item['usage_data']))
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
                    'session_data': json.dumps(session.session_data.model_dump()),
                    'usage_data': json.dumps(session.usage_data.model_dump())
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
                UpdateExpression='SET session_data = :session_data, usage_data = :usage_data',
                ExpressionAttributeValues={
                    ':session_data': json.dumps(session.session_data.model_dump()),
                    ':usage_data': json.dumps(session.usage_data.model_dump())
                }
            )
            return True

        except ClientError:
            self.logger.exception("Error updating session")
            return False
