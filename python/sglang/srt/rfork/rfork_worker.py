import logging
import threading
import time

from sglang.srt.rfork.utils import (
    get_seed,
    release_seed,
    report_seed,
    get_local_seed_key,
)
from sglang.srt.rfork.seed_server import start_rfork_server
from sglang.srt.rfork.transfer_engine import RForkTransferEngineBackendWorker
from sglang.srt.utils.common import get_model_path
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

class RForkWorker:
    def __init__(
            self,
            disaggregation_mode: str,
            node_rank: int,
            tp_rank: int,
            gpu_id: int,
            dtype: str,
            is_draft_model: bool = False,
        ):
        self.disaggregation_mode = disaggregation_mode
        self.node_rank = node_rank
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.dtype = dtype
        self.is_draft_model = is_draft_model
        self.watchdog_event = None
        self.rfork_seed = None
        self.transfer_engine_backend_worker = RForkTransferEngineBackendWorker()
        self.transfer_result = False
        self.ready_to_start_seed_service = False
        self.seed_service_started = False
        self.get_model_path_thread = None
        self.fallback_model_path = None

        self.local_seed_key = get_local_seed_key(
            self.disaggregation_mode,
            self.node_rank,
            self.tp_rank,
            self.is_draft_model,
        )

    def wait_seed_available_or_model_mounted(self):
        def _get_rfork_seed(
            finish_event,
            interval,
        ):
            while not finish_event.is_set():
                rfork_seed = get_seed(
                    self.disaggregation_mode,
                    self.node_rank,
                    self.tp_rank,
                    self.is_draft_model,
                )
                if rfork_seed is not None:
                    self.rfork_seed = rfork_seed
                    break
                time.sleep(interval)
            finish_event.set()

        def _get_model_mounted(
            finish_event,
        ):
            if envs.SGLANG_ASYNC_MODEL_MOUNT.get():
                self.fallback_model_path = get_model_path(with_weights=True)
            finish_event.set()

        finish_event = threading.Event()
        self.get_seed_thread = threading.Thread(
            target=_get_rfork_seed,
            args=(finish_event, 1),
            daemon=True,
            name="RForkGetSeed",
        )
        self.get_model_path_thread = threading.Thread(
            target=_get_model_mounted,
            args=(finish_event,),
            daemon=True,
            name="RForkGetModelMounted",
        )
        self.get_seed_thread.start()
        self.get_model_path_thread.start()
        finish_event.wait()

    def is_seed_available(self) -> bool:
        self.rfork_seed = get_seed(
            self.disaggregation_mode,
            self.node_rank,
            self.tp_rank,
            self.is_draft_model,
        )

        if self.rfork_seed is not None:
            return True

        self.wait_seed_available_or_model_mounted()
        return self.rfork_seed is not None

    def is_transfer_succeeded(self) -> bool:
        return self.transfer_result

    def set_transfer_result(self, result: bool):
        self.transfer_result = result

    def cleanup_after_transfer_failed(self):
        try:
            if not self.ready_to_start_seed_service:
                logger.warning("Not ready to start seed service, no need to cleanup memory regions.")
                return True

            self.ready_to_start_seed_service = False
            assert self.transfer_engine_backend_worker.is_initialized(), "transfer_engine_backend_worker is not initialized, cannot cleanup_after_transfer_failed."
            result = self.transfer_engine_backend_worker.unregister_memory_region()
            return result
        except AssertionError as e:
            logger.exception(f"Unregister memory region failed: {e}")
            return False

    def get_fallback_model_path(self) -> str:
        if self.get_model_path_thread is not None:
            self.get_model_path_thread.join()
        else:
            self.fallback_model_path = get_model_path(with_weights=True)

        return self.fallback_model_path

    def pre_transfer(self, model) -> bool:
        try:
            assert self.transfer_engine_backend_worker.is_initialized(), "transfer_engine_backend_worker is not initialized, cannot pre_transfer."
            result = self.transfer_engine_backend_worker.register_memory_region_v2(model)
            self.ready_to_start_seed_service = result
            return result
        except AssertionError as e:
            logger.exception(f"Pre-transfer failed: {e}")
            return False

    def transfer(self, model) -> bool:
        try:
            assert self.transfer_engine_backend_worker.is_initialized(), "transfer_engine_backend_worker is not initialized, cannot transfer."
            assert self.rfork_seed is not None, "rfork seed is None, cannot transfer."
            return self.transfer_engine_backend_worker.recv_from_source(
                model=model,
                seed_instance_ip=self.rfork_seed["seed_ip"],
                seed_instance_service_port=self.rfork_seed["seed_port"],
                local_seed_key=self.local_seed_key,
            )
        except AssertionError as e:
            logger.exception(f"Transfer failed: {e}")
            return False

    def post_transfer(self):
        if self.rfork_seed is None:
            logger.warning("rfork seed is None, no need to release.")
            return True
        release_seed(self.rfork_seed)
        self.rfork_seed = None
        return True

    def start_seed_service(self, model):
        # Check whether the seed service is already started.
        if self.seed_service_started:
            logger.info("Seed service already started, skipping.")
            return

        # Check whether the instance is ready to serve as seed instance.
        if not self.ready_to_start_seed_service:
            # Not ready, need to prepare for transferring as seed instance first.
            if not self.pre_transfer(model):
                return

        port = start_rfork_server(
            self.local_seed_key,
            (
                self.transfer_engine_backend_worker.rfork_transfer_engine_session_id,
                self.transfer_engine_backend_worker.rfork_transfer_engine_weights_info_dict
            ),
        )
        if port > 0:
            self.rfork_heartbeat_thread = threading.Thread(
                target=report_seed,
                args=(
                    port,
                    self.disaggregation_mode,
                    self.node_rank,
                    self.tp_rank,
                    self.is_draft_model,
                ),
                daemon=True,
                name="RForkHeartbeat",
            )
            self.rfork_heartbeat_thread.start()
            self.seed_service_started = True
