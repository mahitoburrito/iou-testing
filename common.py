from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

from ds.logger import get_logger

log = get_logger()

class Vendor(Enum):
    """ Enum for all annotation vendors """
    CODA = "CODA"
    SCALE = "SCALE"
    AVALA = "AVALA"

class Project(Enum):
    """ Enum for all supported projects """
    SF_BEV = "SF_BEV" 

@dataclass
class S3Path:
    bucket_name: str
    key: str  # For directories, this should end with '/'
    size: int = -1  # -1 for unknown, -2 when known to be a directory
    storage_class: Optional[str] = None
    archive_status: Optional[str] = None
    mtime: datetime = None  # Time when this file was last modified. Note: S3 stores time at seconds level precision
    e_tag: Optional[str] = None  # Etag value from S3

    def __post_init__(self):
        self.key = self.key.rstrip('/')  # Remove any trailing slashes

    @property
    def arn(self) -> str:
        """ Returns this object's ARN """
        return f'arn:aws:s3:::{self.bucket_name}/{self.key}'

    @property
    def arn_bucket(self) -> str:
        """ Returns bucket's ARN """
        return f'arn:aws:s3:::{self.bucket_name}'

    @property
    def name(self) -> str:
        return self.key.split('/')[-1]

    @classmethod
    def from_s3uri(cls, s3uri: str) -> "S3Path":
        """
        Ctor. Returns class object from the S3 URI.
        :param s3uri: String. E.g. "s3://bucket_name/path/to/file.txt"
        :return: An object of class S3Path
        """
        assert len(s3uri) > 0  # Sanity check
        parsed_url = urlparse(s3uri)
        bucket_name = parsed_url.netloc  # Extract bucket name out of s3 URL
        path_within_bucket = parsed_url.path.lstrip('/')  # Remove the leading forward slash
        return cls(bucket_name, path_within_bucket)

    def copy(self) -> "S3Path":
        return S3Path(bucket_name=self.bucket_name, key=self.key, size=self.size)

    def get_parent(self) -> "S3Path":
        # S3Path of parent directory
        return S3Path(bucket_name=self.bucket_name, key='/'.join(self.key.split('/')[:-1]), size=-2)

    def get_parent_s3uri(self) -> str:
        return '/'.join(self.key.split('/')[:-1])  # S3 URI of parent directory

    def to_uri(self) -> str:
        return f"s3://{self.bucket_name}/{self.key}"

    def __repr__(self):
        return self.to_uri()

    def extend_path(self, *args) -> "S3Path":
        components = [self.key] + list(args)
        components = [x.strip('/') for x in components]
        if components[-1].endswith('/'):  # In case path_extn pointed to a directory, add '/' suffix to output
            components.append('')

        return S3Path(self.bucket_name, '/'.join(components))


@dataclass
class HpcPath:
    """ Class stores an object's path in s3 and HPC """
    s3: S3Path  # Path on S3. E.g "s3://bucket/path/to/object.txt"
    hpc: Optional[Path] = None  # Path on HPC. E.g. "/perception_data/lucid_data/batch_123/object.txt"

    @classmethod
    def from_s3_path(cls, s3_path: S3Path, hpc_dir: Path, session_name: str) -> "HpcPath":
        """ Returns HpcPath for a session's object from s3_path and HPC """
        idx: int = s3_path.key.find(session_name)  # Find the location of session name in S3 key
        session_key_len: int = idx + len(session_name) + 1  # Find the location where session name ends in S3 key

        s3_path_split = s3_path.key[session_key_len:].split('/')
        hpc_path = hpc_dir.joinpath(*s3_path_split)  # Path in HPC

        return HpcPath(s3=s3_path, hpc=hpc_path)

    @classmethod
    def from_s3uri(cls, s3_uri: str, hpc_dir: Path, session_name: str) -> "HpcPath":
        """ Returns HpcPath for a session's object from S3 uri and HPC """
        idx: int = s3_uri.find(session_name)  # Find the location of session name in S3 key
        session_key_len: int = idx + len(session_name) + 1  # Find the location where session name ends in S3 key

        s3_path_split = s3_uri[session_key_len:].split('/')
        hpc_path = hpc_dir.joinpath(*s3_path_split)  # Path in HPC

        return HpcPath(s3=S3Path.from_s3uri(s3_uri), hpc=hpc_path)

    def to_hpc_dict(self) -> Dict:
        return {"s3": self.s3.to_uri(), "hpc": str(self.hpc), "size": self.s3.size,
                "mtime": self.s3.mtime.isoformat(), "etag": self.s3.e_tag}
