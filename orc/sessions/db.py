import json
import logging
from datetime import datetime
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from orc.sessions.models import SessionItem, UsageData


class DynamoDBSessionManager:

    def __init__(self, table_name: str):
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table(table_name)

        self.logger = logging.getLogger()
        self.logger.setLevel("INFO")

    @staticmethod
    def _serialize_datetime(dt: datetime) -> str:
        return dt.isoformat() if dt else None

    @staticmethod
    def _deserialize_datetime(dt_str: str) -> datetime:
        return datetime.fromisoformat(dt_str) if dt_str else None

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
                    created_at=self._deserialize_datetime(item['created_at']),
                    usage_data=UsageData(**item['usage_data']) if 'usage_data' in item else None
                )
            return None

        except ClientError:
            self.logger.exception("Error getting session")
            return None

    def create_session(self, user_id: str, session_id: str) -> bool:
        try:
            self.table.put_item(
                Item={
                    'user_id': user_id,
                    'session_id': session_id,
                    'created_at': self._serialize_datetime(datetime.utcnow()),
                    'usage_data': json.dumps(UsageData().model_dump())
                },
                ConditionExpression='attribute_not_exists(user_id) AND attribute_not_exists(session_id)'
            )
            return True

        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                self.logger.error(f"Session already exists for user {user_id} and session {session_id}")
            else:
                self.logger.exception(f"Error creating session")
            return False

    def update_session_usage(self, user_id: str, session_id: str, usage: UsageData) -> bool:
        try:
            self.table.update_item(
                Key={
                    'user_id': user_id,
                    'session_id': session_id
                },
                UpdateExpression='SET usage_data = :usage_data',
                ExpressionAttributeValues={
                    ':usage_data': json.dumps(usage.model_dump())
                }
            )
            return True

        except ClientError:
            self.logger.exception("Error updating session")
            return False
