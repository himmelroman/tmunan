import os
import json
import urllib.request
import logging
from jose import jwt
from jose.exceptions import JWTError, ExpiredSignatureError, JWTClaimsError

# Set up the logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

ALGORITHMS = ["RS256"]


def get_auth0_domain():
    """Fetch Auth0 domain from environment variables."""
    return os.environ.get("AUTH0_DOMAIN")


def get_api_audience():
    """Fetch API audience from environment variables."""
    return os.environ.get("API_AUDIENCE")


def get_jwks():
    """Fetch JWKS keys from Auth0."""
    auth0_domain = get_auth0_domain()
    jwks_url = f"https://{auth0_domain}/.well-known/jwks.json"
    logger.info(f"Fetching JWKS from: {jwks_url}")

    with urllib.request.urlopen(jwks_url) as response:
        return json.load(response)


def get_signing_key(jwks, token):
    """Fetch signing key from JWKS using the token's header."""
    unverified_header = jwt.get_unverified_header(token)
    logger.debug(f"Unverified header: {unverified_header}")

    for key in jwks["keys"]:
        if key["kid"] == unverified_header["kid"]:
            logger.info(f"Found matching signing key for kid: {unverified_header['kid']}")
            return {
                "kty": key["kty"],
                "kid": key["kid"],
                "use": key["use"],
                "n": key["n"],
                "e": key["e"],
            }
    raise Exception("Unable to find appropriate key")


def verify_token(token):
    """Decode and verify the JWT."""
    try:
        # Fetch JWKS and the signing key
        jwks = get_jwks()
        rsa_key = get_signing_key(jwks, token)

        if not rsa_key:
            raise Exception("No valid signing key found")

        # Verify the JWT signature and claims
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=ALGORITHMS,
            audience=get_api_audience(),
            issuer=f"https://{get_auth0_domain()}/"
        )
        logger.info("Token successfully verified")
        return payload

    except ExpiredSignatureError:
        logger.error("Token has expired")
        raise Exception("Token has expired")
    except JWTClaimsError as e:
        logger.error(f"JWT Claims error: {str(e)}")
        raise Exception(f"JWT Claims error: {str(e)}")
    except JWTError as e:
        logger.error(f"JWT Error: {str(e)}")
        raise Exception(f"JWT Error: {str(e)}")
