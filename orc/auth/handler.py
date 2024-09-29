import logging

from orc.auth.token import verify_token
from orc.auth.policy import generate_policy

# Set up the logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """AWS Lambda Authorizer handler."""
    try:
        # Extract the token from the event (API Gateway authorizer setup)
        token = event.get("authorizationToken", "").split(" ")[1]
        logger.info(f"Received token: {token}")

        # Validate and decode the token
        payload = verify_token(token)
        logger.info(f"Received payload: {payload}")

        # Generate an allow policy and pass claims in the context
        return {
            "principalId": payload["sub"],  # User's unique identifier from the token
            "policyDocument": generate_policy("user", "Allow", event["methodArn"]),
            "context": {
                "user_id": payload["sub"],              # Example claim
                "email": payload.get("email", None),    # Optional, if available in token
                "name": payload.get("name", None)       # Optional
            }
        }

    except Exception as e:

        # Log the error
        logger.exception(f"Error in auth lambda handler!")

        # return a deny policy
        return {
            "principalId": "user",
            "policyDocument": generate_policy("user", "Deny", event["methodArn"])
        }
