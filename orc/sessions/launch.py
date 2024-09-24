import os

import boto3
import logging

logger = logging.getLogger()
logger.setLevel("INFO")

ecs_client = boto3.client('ecs')


def launch_session(user_id, session_id, signaling_channel):

    # read ECS vars
    cluster_name = os.getenv('ECS_CLUSTER_NAME')
    task_definition = os.getenv('TASK_DEFINITION')

    # log
    logging.info(f'Run Session - Parameters: {signaling_channel=}')

    # verify params
    if not cluster_name or not task_definition or not signaling_channel:
        raise Exception('Required parameters missing')

    else:
        logging.info(f'Launch Session - Launching task: {cluster_name=}, {task_definition=}')

        # launch task on ECS
        response = ecs_client.run_task(
            cluster=cluster_name,
            taskDefinition=task_definition,
            count=1,
            overrides={
                'containerOverrides': [
                    {
                        'name': 'stream',
                        'environment': [
                            {
                                'name': 'USER_ID',
                                'value': user_id
                            },
                            {
                                'name': 'SESSION_ID',
                                'value': session_id
                            },
                            {
                                'name': 'SIGNALING_CHANNEL',
                                'value': signaling_channel
                            }
                        ]
                    }
                ]
            }
        )

        # log response
        logging.info(f'Launch Session - ECS Response: {response}')

        # prepare return info
        return {
                'cluster_arn': response['tasks'][0]['clusterArn'],
                'task_arn': response['tasks'][0]['taskArn']
            }
