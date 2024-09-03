import logging
import os
import json
import boto3

ecs_client = boto3.client('ecs')


def run_session(event, context):

    # read ECS vars
    cluster_name = os.getenv('ECS_CLUSTER_NAME')
    task_definition = os.getenv('TASK_DEFINITION')

    # read session vars
    signaling_channel = event['queryStringParameters'].get('signaling_channel', None)

    # log
    logging.info(f'Run Session - Parameters: {signaling_channel=}')

    # verify params
    if cluster_name and task_definition and signaling_channel:

        logging.info(f'Run Session - Launching task: {cluster_name=}, {task_definition=}')

        # launch task on ECS
        container_name = 'stream'
        response = ecs_client.run_task(
            cluster=cluster_name,
            taskDefinition=task_definition,
            count=1,
            overrides={
                'containerOverrides': [
                    {
                        'name': container_name,
                        'environment': [
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
        logging.info(f'Run Session - ECS Response: {response}')

        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'success'
            })
        }

    else:

        return {
            'statusCode': 400,
            'body': json.dumps({
                'error': 'Required parameters missing'
            })
        }