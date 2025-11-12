import logging
import os
import shutil
import tarfile
from oss2 import Auth, Bucket
from oss2.exceptions import NoSuchKey
from pathlib import Path
from typing import List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

MODEL_URL = os.getenv("MODEL_URL", "")
MODEL_NAME = os.path.basename(urlparse(MODEL_URL).path)
MODEL_DEPLOY_STRATEGY_NAME = os.getenv("MODEL_DEPLOY_STRATEGY_NAME")
MODEL_INSTANCE_ROLE = os.getenv("MODEL_INSTANCE_ROLE", "default")
NODE_RANK = os.getenv("NODE_RANK", "0")
OSS_ENDPOINT = os.getenv("COMPILE_CACHE_OSS_ENDPOINT")
OSS_BUCKET = os.getenv("COMPILE_CACHE_OSS_BUCKET")
ACCESS_KEY_ID = os.getenv("COMPILE_CACHE_OSS_ACCESS_KEY_ID")
ACCESS_KEY_SECRET = os.getenv("COMPILE_CACHE_OSS_ACCESS_KEY_SECRET")
COMPILE_CACHE_OSS_PREFIX = os.getenv("COMPILE_CACHE_OSS_PREFIX")

OSS_PREFIX = (
    f"{MODEL_NAME}-{MODEL_DEPLOY_STRATEGY_NAME}-{MODEL_INSTANCE_ROLE}-{NODE_RANK}"
)

TEMP_DIR = "/tmp/compile_cache"


def _upload_to_oss(
    local_file: str,
    endpoint: str,
    access_key_id: str,
    access_key_secret: str,
    bucket_name: str,
    oss_key: str,
) -> bool:
    if not os.path.isfile(local_file):
        logger.error(f"file not exists: {local_file}")
        return False

    try:
        auth = Auth(access_key_id, access_key_secret)
        bucket = Bucket(auth, endpoint, bucket_name)

        bucket.put_object_from_file(oss_key, local_file)
        logger.debug(f"finish uploading: {local_file} -> oss://{bucket_name}/{oss_key}")
        return True
    except Exception as e:
        logger.error(f"OSS upload failed: {e}")
        return False


def _download_from_oss(
    local_file: str,
    endpoint: str,
    access_key_id: str,
    access_key_secret: str,
    bucket_name: str,
    oss_key: str,
) -> bool:
    try:
        auth = Auth(access_key_id, access_key_secret)
        bucket = Bucket(auth, endpoint, bucket_name)

        os.makedirs(os.path.dirname(local_file), exist_ok=True)

        bucket.get_object_to_file(oss_key, local_file)
        logger.debug(
            f"finish downloading: oss://{bucket_name}/{oss_key} -> {local_file}"
        )
        return True
    except NoSuchKey:
        logger.info(f"Cache miss: {oss_key}")
        return False
    except Exception as e:
        logger.error(f"OSS download failed: {e}")
        return False


def _compress_to_targz(source_dir: str, output_tar_gz: str, filter=None) -> bool:
    source_path = Path(source_dir)
    if not source_path.exists() or not source_path.is_dir():
        logger.error(f"path not exists or is not dir: {source_dir}")
        return False

    try:
        with tarfile.open(output_tar_gz, "w:gz") as tar:
            tar.add(source_path, arcname=source_path.name, filter=filter)
        logger.debug(f"compress succeeded: {output_tar_gz}")
        return True
    except Exception as e:
        logger.error(f"compress {source_dir} -> {output_tar_gz} failed: {e}")
        return False


def _decompress_targz(tar_gz_file: str, extract_to: str) -> bool:
    if not Path(tar_gz_file).exists():
        logger.error(f"file not exists: {tar_gz_file}")
        return False

    try:
        with tarfile.open(tar_gz_file, "r:gz") as tar:
            tar.extractall(path=extract_to, filter="fully_trusted")
        logger.debug(f"decompress succeeded: {tar_gz_file} -> {extract_to}")
        return True
    except Exception as e:
        logger.error(f"decompress failed: {e}")
        return False


def _check_env() -> bool:
    env_vars = {
        "MODEL_URL": MODEL_URL,
        "MODEL_DEPLOY_STRATEGY_NAME": MODEL_DEPLOY_STRATEGY_NAME,
        "OSS_ENDPOINT": OSS_ENDPOINT,
        "OSS_BUCKET": OSS_BUCKET,
        "ACCESS_KEY_ID": ACCESS_KEY_ID,
        "ACCESS_KEY_SECRET": ACCESS_KEY_SECRET,
        "COMPILE_CACHE_OSS_PREFIX": COMPILE_CACHE_OSS_PREFIX,
    }

    missing = []
    for name, value in env_vars.items():
        if value is None or value == "":
            missing.append(name)

    if missing:
        logger.error(
            "The following environment variables for compile cache are unset: %s",
            ", ".join(missing),
        )
        return False
    else:
        return True


def prepare_compile_cache(cache_root: str, cache_dirs: List):
    if not _check_env():
        return

    os.makedirs(TEMP_DIR, exist_ok=True)

    for name in cache_dirs:
        local_tar_gz = os.path.join(TEMP_DIR, f"{name}.tar.gz")
        oss_key = f"{COMPILE_CACHE_OSS_PREFIX}/{OSS_PREFIX}/{name}.tar.gz"

        logger.info(f"downloading cache of {name}")
        if _download_from_oss(
            local_file=local_tar_gz,
            endpoint=OSS_ENDPOINT,
            access_key_id=ACCESS_KEY_ID,
            access_key_secret=ACCESS_KEY_SECRET,
            bucket_name=OSS_BUCKET,
            oss_key=oss_key,
        ) and _decompress_targz(local_tar_gz, cache_root):
            logger.info(f"finish loading {name}")

    shutil.rmtree(TEMP_DIR)


def save_compile_cache(cache_root: str, cache_dirs: List):
    if not _check_env():
        return

    os.makedirs(TEMP_DIR, exist_ok=True)

    for name in cache_dirs:
        source_dir = os.path.join(cache_root, name)
        tar_gz_file = os.path.join(TEMP_DIR, f"{name}.tar.gz")
        oss_key = f"{COMPILE_CACHE_OSS_PREFIX}/{OSS_PREFIX}/{name}.tar.gz"

        logger.info(f"try to upload cache of {name}")

        if _compress_to_targz(source_dir, tar_gz_file) and _upload_to_oss(
            local_file=tar_gz_file,
            endpoint=OSS_ENDPOINT,
            access_key_id=ACCESS_KEY_ID,
            access_key_secret=ACCESS_KEY_SECRET,
            bucket_name=OSS_BUCKET,
            oss_key=oss_key,
        ):
            logger.info(f"finish uploading cache of {name}")

    shutil.rmtree(TEMP_DIR)
