import requests
import socket
import logging
import time
import importlib
import importlib.util
import os

from urllib.error import HTTPError

logger = logging.getLogger(__name__)


def _get_scheduler_url():
    url = os.getenv("RFORK_SCHEDULER_URL", None)
    if url is None:
        raise RuntimeError("RFORK_SCHEDULER_URL is not set. Cannot interact with the scheduler.")
    return url


def _get_seed_key():
    model_url = os.getenv("MODEL_URL", None)
    strategy = os.getenv("MODEL_DEPLOY_STRATEGY_NAME", None)
    sep = os.getenv("RFORK_SEED_KEY_SEPERATOR", "$")
    if model_url is None or strategy is None:
        raise RuntimeError("SEED_KEY is not set. Ensure ENV MODEL_URL and MODEL_DEPLOY_STRATEGY_NAME are set.")
    return f"{model_url}{sep}{strategy}"


def _get_seed_key_seperator():
    return os.getenv("RFORK_SEED_KEY_SEPERATOR", "$")


def _ensure_scheduler_url_set():
    _get_scheduler_url()


def _ensure_seed_key_set():
    _get_seed_key()

def is_transfer_engine_supported():
    return importlib.util.find_spec("mooncake.engine") is not None

def get_seed(disaggregation_mode: str, node_rank: int, tp_rank: int, is_draft_worker: bool = False):
    try:
        _ensure_scheduler_url_set()

        seed_key = get_local_seed_key(
            disaggregation_mode,
            node_rank,
            tp_rank,
            is_draft_worker,
        )

        response = requests.get(
            f"{_get_scheduler_url()}/get_seed",
            headers={
                "SEED_KEY": seed_key,
            },
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get seed from the planner, {response.status_code}")

        seed_ip = response.headers.get("SEED_IP")
        seed_port = response.headers.get("SEED_PORT")
        user_id = response.headers.get("USER_ID")
        seed_rank = response.headers.get("SEED_RANK")
        if seed_ip is None or seed_port is None or user_id is None or seed_rank is None:
            raise RuntimeError(
                f"Failed to parse valid seed info, at least one necessary info is None. \
                seed_ip: {seed_ip}, seed_port: {seed_port}, user_id: {user_id}, seed_rank: {seed_rank}"
            )

        logger.debug(
            f"seed_ip: {seed_ip}, seed_port: {seed_port}, user_id: {user_id}, seed_rank: {seed_rank}"
        )
        return {
            "seed_ip": seed_ip,
            "seed_port": seed_port,
            "user_id": user_id,
            "seed_rank": seed_rank,
        }

    except RuntimeError as e:
        logger.error(f"get_seed from planner RuntimeError: {e}")
        return None
    except HTTPError as e:
        logger.exception(f"get_seed from planner HTTPError: {e}")
        return None
    except Exception as e:
        logger.exception(f"get_seed from planner Exception: {e}")
        return None

def release_seed(seed) -> bool:
    try:
        _ensure_scheduler_url_set()
        user_id = seed["user_id"]
        seed_ip = seed["seed_ip"]
        seed_port = str(seed["seed_port"])
        seed_rank = str(seed["seed_rank"])

        response = requests.post(
            f"{_get_scheduler_url()}/put_seed",
            headers={
                "SEED_IP": seed_ip,
                "SEED_PORT": seed_port,
                "USER_ID": user_id,
                "SEED_RANK": seed_rank,
            },
        )

        if response.status_code != 200:
            raise RuntimeError(f"Failed to release seed to the planner, {response.status_code}")

        return True
    except RuntimeError as e:
        logger.exception(f"release_seed to planner RuntimeError: {e}")
        return False
    except HTTPError as e:
        logger.exception(f"release_seed to planner HTTPError: {e}")
        return False
    except Exception as e:
        logger.exception(f"release_seed to planner Exception: {e}")
        return False

def report_seed(
    port: int,
    disaggregation_mode: str,
    node_rank: int,
    tp_rank: int,
    is_draft_worker: bool = False,
    sleep_interval: int = 30,
):
    seed_key = None
    try:
        _ensure_scheduler_url_set()

        seed_ip = socket.gethostbyname(socket.gethostname())

        seed_key = get_local_seed_key(
            disaggregation_mode,
            node_rank,
            tp_rank,
            is_draft_worker,
        )
    except Exception as e:
        logger.exception(f"report_seed setup Exception: {e}")
        return

    while True:
        result = False
        try:
            response = requests.post(
                f"{_get_scheduler_url()}/add_seed",
                headers={
                    "SEED_KEY": seed_key,
                    "SEED_IP": seed_ip,
                    "SEED_PORT": str(port),
                    "SEED_RANK": str(tp_rank),
                    "SEED_REFCNT": str(0),
                },
            )
            if response.status_code == 200:
                result = True
        except HTTPError as e:
            logger.exception(f"report_seed to planner HTTPError: {e}")
        except Exception as e:
            logger.exception(f"report_seed to planner Exception: {e}")

        logger.info(f"[rfork_heartbeat] report seed to planner result: {result}")
        time.sleep(sleep_interval)

def get_local_seed_key(
    disaggregation_mode: str,
    node_rank: int,
    tp_rank: int,
    is_draft_worker: bool = False,
) -> str:
    _ensure_seed_key_set()

    sep = _get_seed_key_seperator()
    seed_key = _get_seed_key()

    key_suffix = f"{disaggregation_mode}{sep}{node_rank}{sep}{tp_rank}"
    if is_draft_worker:
        key_suffix += f"{sep}draft"
    seed_key = f"{seed_key}{sep}{key_suffix}"

    return seed_key
