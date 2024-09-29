import logging

# Set up the logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def generate_policy(principal_id, effect, resource):
    """Generate an IAM policy for API Gateway."""
    logger.info(f"Generating policy for principal_id: {principal_id}, effect: {effect}, resource: {resource}")

    auth_response = {
        "principalId": principal_id
    }

    if effect and resource:
        auth_response["policyDocument"] = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": "execute-api:Invoke",
                    "Effect": effect,
                    "Resource": resource
                }
            ]
        }

    return auth_response
