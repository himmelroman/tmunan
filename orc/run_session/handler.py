import json
import boto3
import os

ecs_client = boto3.client('ecs')


def run_session(event, context):

    # read ECS vars
    cluster_name = os.getenv('ECS_CLUSTER_NAME')
    task_definition = os.getenv('TASK_DEFINITION')

    # read session vars
    signaling_channel = event['queryStringParameters'].get('SIGNALING_CHANNEL')

    # launch task on ECS
    container_name = 'stream'
    response = ecs_client.run_task(
        cluster=cluster_name,
        taskDefinition=task_definition,
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
        },
        count=1,
        launchType='EC2'
    )

    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }
