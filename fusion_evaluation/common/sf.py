import os
import json
import botocore
import botocore.session
from aws_secretsmanager_caching import SecretCache, SecretCacheConfig


def fetch_snowflake_config(secret_name: str = 'prod/rakuten_rewards/snowflake/RI_CONSULTING_ADMIN'):
    """Retrieves a snowflake configuration from Secrets Manager by Secret name

    Args:
        secret_name (str): Name of the secret in AWS Secrets Manager
            (default is the production config for RI_CONSULTING_ADMIN)

    Returns:
        dict: Snowflake configuration including account, user, password, role, warehouse
            (not including database and schema)

    """

    # This will usually use a cache, and auto-magically refetch secrets when they go stale or bad
    secrets = SecretCache(
        config=SecretCacheConfig(),  # all defaults
        client=botocore.session.get_session().create_client('secretsmanager')
    )

    # Snowflake secrets will eventually be provided by the automation pipeline at SSM_SECRET_NAME_SNOWFLAKE_CREDS
    snowflake_secret_name = os.getenv(
        'SSM_SECRET_NAME_SNOWFLAKE_CREDS',
        default=secret_name
    )

    # get secrets from cache and parse
    cached_secret = secrets.get_secret_string(snowflake_secret_name)
    parsed_secret = json.loads(cached_secret)

    return parsed_secret
