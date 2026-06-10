import requests
import logging
import time
import torch

from sglang.srt.utils.network import get_local_ip_auto
from sglang.srt.rfork.utils import is_transfer_engine_supported

logger = logging.getLogger(__name__)

class RForkTransferEngineBackendWorker:
    def __init__(self):
        self.rfork_transfer_engine = None
        self.rfork_transfer_engine_session_id = None
        self.rfork_transfer_engine_weights_info_dict = None
        self.registered_weight_blocks = []
        self._is_initialized = False
        if is_transfer_engine_supported():
            self.init_transfer_engine()

    def init_transfer_engine(self):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            raise ImportError(
                "Please install mooncake for rfork transfer engine: pip install mooncake"
            ) from e
        self.rfork_transfer_engine = TransferEngine()
        local_ip = get_local_ip_auto()
        self.rfork_transfer_engine.initialize(
            local_ip,
            "P2PHANDSHAKE",
            "rdma",
            "" # auto discovery if empty
        )
        self.rfork_transfer_engine_session_id = f"{local_ip}:{self.rfork_transfer_engine.get_rpc_port()}"
        self._is_initialized = True

    def is_initialized(self) -> bool:
        return self._is_initialized

    def register_memory_region(self, model):
        if self.rfork_transfer_engine_weights_info_dict is not None:
            logger.warning("Memory region has already been registered.")
            return True
        if self.rfork_transfer_engine is None:
            logger.error("rfork_transfer_engine is not initialized.")
            return False
        start_reg_mr_tic = time.time()
        weight_mr_dict = {}
        for name, weight in model.named_parameters():
            ret = self.rfork_transfer_engine.register_memory(weight.data_ptr(), weight.numel() * weight.element_size())
            if ret != 0:
                logger.error(f"register memory failed for weight {name}, error: {ret}")
                return False
            weight_mr_dict[name] = (weight.data_ptr(), weight.numel(), weight.element_size())
        self.rfork_transfer_engine_weights_info_dict = weight_mr_dict
        end_reg_mr_tic = time.time()
        logger.warning(f"register memory region time: {(end_reg_mr_tic - start_reg_mr_tic):.4f}s")
        return True

    def register_memory_region_v2(self, model):
        start_reg_mr_tic = time.time()

        weight_mr_dict = {}
        weight_addr_set = set()
        for name, weight in model.named_parameters():
            weight_mr_dict[name] = (weight.data_ptr(), weight.numel(), weight.element_size())
            weight_addr_set.add(weight.data_ptr())

        # Get memory snapshot.
        memory_snapshot = torch.cuda.memory.memory_snapshot()
        weight_blocks_for_reg_mr = []
        for segment in memory_snapshot:
            current_weight_block = None
            blocks = segment.get("blocks", [])
            for block in blocks:
                address = block.get("address", -1)
                size = block.get("size", -1)
                state = block.get("state", "")
                if address < 0 or size < 0 or state == "":
                    continue
                if state == "active_allocated":
                    if address in weight_addr_set:
                        if current_weight_block is None:
                            current_weight_block = (address, size)
                        elif current_weight_block[0] + current_weight_block[1] == address:
                            current_weight_block = (current_weight_block[0], current_weight_block[1] + size)
                        else:
                            weight_blocks_for_reg_mr.append(current_weight_block)
                            current_weight_block = (address, size)
            if current_weight_block is not None:
                weight_blocks_for_reg_mr.append(current_weight_block)

        registered_block_count = 0
        for weight_block in weight_blocks_for_reg_mr:
            address, size = weight_block
            ret = self.rfork_transfer_engine.register_memory(address, size)
            if ret != 0:
                logger.error(f"register_memory_region_v2 failed for address {address}, size {size}, error: {ret}")
                for i in range(0, registered_block_count):
                    self.rfork_transfer_engine.unregister_memory(weight_blocks_for_reg_mr[i][0])
                return False
            registered_block_count += 1

        self.rfork_transfer_engine_weights_info_dict = weight_mr_dict
        self.registered_weight_blocks = weight_blocks_for_reg_mr

        end_reg_mr_tic = time.time()
        logger.warning(f"register_memory_region_v2 time: {(end_reg_mr_tic - start_reg_mr_tic):.4f}s")
        return True

    def unregister_memory_region(self) -> bool:
        start_unreg_mr_tic = time.time()
        for weight_block in self.registered_weight_blocks:
            address, _ = weight_block
            ret = self.rfork_transfer_engine.unregister_memory(address)
            if ret != 0:
                logger.error(f"unregister memory failed for address {address}, error: {ret}")
                return False
        self.rfork_transfer_engine_weights_info_dict = None
        self.registered_weight_blocks = []
        end_unreg_mr_tic = time.time()
        logger.warning(f"unregister_memory_region time: {(end_unreg_mr_tic - start_unreg_mr_tic):.4f}s")
        return True

    def recv_from_source(self, model, seed_instance_ip, seed_instance_service_port, local_seed_key):
        seed_url = f"http://{seed_instance_ip}:{seed_instance_service_port}"
        seed_transfer_engine_session_id, seed_transfer_engine_weight_info = get_remote_instance_transfer_engine_info(seed_url, local_seed_key)
        if seed_transfer_engine_session_id is None or seed_transfer_engine_weight_info is None:
            logger.error("Cannot get transfer engine session or weight info.")
            return False
        seed_ptr_list = []
        client_ptr_list = []
        client_len_list = []
        for name, tensor in model.named_parameters():
            weight_info = seed_transfer_engine_weight_info.get(name, None)
            if weight_info is None:
                logger.error(f"Cannot find weight info for {name}.")
                return False

            seed_ptr, seed_len, seed_size = weight_info
            if seed_len != tensor.numel() or seed_size != tensor.element_size():
                logger.error(
                    f"Weight info does not match for {name}, "
                    f"expected ({seed_len}, {seed_size}), "
                    f"got ({tensor.numel()}, {tensor.element_size()})"
                )
                return False
            client_ptr = tensor.data_ptr()
            client_len = tensor.numel() * tensor.element_size()
            seed_ptr_list.append(seed_ptr)
            client_ptr_list.append(client_ptr)
            client_len_list.append(client_len)
        logger.warning("start transferring weights from remote instance RDMA ...")
        start_transfer_tic = time.time()
        ret = self.rfork_transfer_engine.batch_transfer_sync_read(
            seed_transfer_engine_session_id,
            client_ptr_list,
            seed_ptr_list,
            client_len_list
        )
        if ret < 0:
            logger.error("Failed to transfer weights from remote instance.")
            return False
        end_transfer_tic = time.time()
        logger.warning(f"transfer weights time: {(end_transfer_tic - start_transfer_tic):.4f}s")
        return True

def get_remote_instance_transfer_engine_info(seed_url: str, local_seed_key: str):
    try:
        response = requests.get(
            f"{seed_url}/get_rfork_transfer_engine_info",
            params={"seed_key": local_seed_key},
        )

        if response.status_code == 200:
            data = response.json()

            rfork_transfer_engine_info = data.get('rfork_transfer_engine_info', None)
            if (
                rfork_transfer_engine_info is not None
                and isinstance(rfork_transfer_engine_info, list)
                and len(rfork_transfer_engine_info) == 2
            ):
                return rfork_transfer_engine_info[0], rfork_transfer_engine_info[1]
            else:
                logger.error("Failed to get `rfork_transfer_engine_info` in response.")
                return None, None
        else:
            logger.error(f"request.get failed: {response.status_code}")
            return None, None
    except Exception as e:
        logger.error(f"Exception: {e}")
        return None, None
