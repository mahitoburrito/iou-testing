import os
from pathlib import Path
from typing import Optional

import addict
import boto3
import botocore.exceptions
import yaml

import ds.errors as e
from ds.logger import get_logger
from ds.common import S3Path

log = get_logger()


def _read_yaml(file: str) -> dict:
    """
    Reads a Yaml file
    :param file: String. Path to the local yaml file
    :return: A Python dict
    """
    try:
        with open(file, 'r') as fr:
            yaml_info = yaml.load(fr, yaml.Loader)
            return yaml_info
    except Exception as exc:
        raise e.BadCfgFile(f"Error while parsing config params from file: {file} {exc}") from None


def _update_params_from_env(cfg):
    # DB credentials
    if all([cfg.database,
            os.getenv('PG_TIMESERIES_DB_NAME', None), os.getenv('PG_TIMESERIES_PORT', None),
            os.getenv('PG_TIMESERIES_HOST', None), os.getenv('PG_TIMESERIES_USERNAME', None),
            os.getenv('PG_TIMESERIES_PASSWORD', None)]):
        log.debug('Getting DB credentials from env vars ...')
        cfg.database.from_env = True
        cfg.database.db_name = os.getenv('PG_TIMESERIES_DB_NAME', None)
        cfg.database.db_host = os.getenv('PG_TIMESERIES_HOST', None)
        cfg.database.db_port = os.getenv('PG_TIMESERIES_PORT', None)
        cfg.database.db_uname = os.getenv('PG_TIMESERIES_USERNAME', None)
        cfg.database.db_paswd = os.getenv('PG_TIMESERIES_PASSWORD', None)

        if cfg.database.db_port is not None:  # Convert from string to int
            cfg.database.db_port = int(cfg.database.db_port)
    else:
        log.warning('DB credentials were not found in env vars ...')
        cfg.database.from_env = False  # Mark that we didn't pick db credentials from environment

    return


def get_config(path_cfg: str) -> addict.Dict:
    """
    Parses the configuration file and returns the configuration object,
    :return: Configuration object of addict.Dict class.
    """
    if not os.path.exists(path_cfg):
        raise e.BadCfgFile(f'Expected config file not found: {path_cfg}')

    cfg = addict.Dict(_read_yaml(path_cfg))

    # Backup, hide, display and then restore sensitive data
    db_uname, cfg.database.db_uname = cfg.database.db_uname, 'xxxx'
    db_paswd, cfg.database.db_paswd = cfg.database.db_paswd, 'xxxx'
    db_port, cfg.database.db_port = cfg.database.db_port, 'xxxx'
    log.debug('Configuration parameters: {}'.format(str(cfg)))  # Display config parameters
    # Restore sensitive data
    cfg.database.db_uname = db_uname
    cfg.database.db_paswd = db_paswd
    cfg.database.db_port = db_port

    _update_params_from_env(cfg)  # Replace some confidential values from env vars

    # Convert to correct types
    cfg.s3_root = S3Path.from_s3uri(str(cfg.s3_root))

    return cfg

def get_aws_session(region_name: str, profile_name: Optional[str]) -> boto3.Session:
    """ Returns a new AWS session """
    if profile_name is not None:
        log.warning(f'Expecting ~/.aws/credentials to have credentials for profile: {profile_name}')

    try:
        aws_session = boto3.Session(profile_name=profile_name, region_name=region_name)
    except botocore.exceptions.ClientError as exc:
        raise e.AwsError(f"Failed to create AWS session for {profile_name} {region_name}: {exc}") from None

    return aws_session
