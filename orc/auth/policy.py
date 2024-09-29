import logging

# Set up the logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def generate_policy(effect, resource):
    """Generate an IAM policy for API Gateway."""

    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Action": "execute-api:Invoke",
                "Effect": effect,
                "Resource": resource
            }
        ]
    }
