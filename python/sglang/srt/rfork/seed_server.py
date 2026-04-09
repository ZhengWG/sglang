import threading
import time
import requests
from fastapi import FastAPI
from fastapi.responses import Response
from http import HTTPStatus
import uvicorn
import socket
import logging
import queue

logger = logging.getLogger(__name__)

def start_fastapi_server(queue, local_seed_key, info):
    logger.warning("[Child] Preparing socket with dynamic port...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", 0))
    _, port = sock.getsockname()
    logger.warning(f"[Child] Assigned dynamic port: {port}")

    app = FastAPI()

    @app.get("/get_rfork_transfer_engine_info")
    def get_rfork_transfer_engine_info(seed_key: str):
        if seed_key == local_seed_key:
            return { "rfork_transfer_engine_info": info }
        else:
            return { "rfork_transfer_engine_info": None }

    @app.get("/rfork_fetch_seed")
    def rfork_fetch_seed():
        return { "status": "ok" }

    @app.get("/health_check_with_key")
    def health_check_with_key(seed_key: str):
        if seed_key == local_seed_key:
            return Response(status_code=HTTPStatus.OK)
        else:
            return Response(status_code=HTTPStatus.BAD_REQUEST)

    config = uvicorn.Config(app, host=None, port=None, log_level="warning")
    server = uvicorn.Server(config)

    try:
        queue.put(port)
    except Exception as e:
        logger.error(f"[Child] Failed to send port via queue: {e}")
        sock.close()
        return

    logger.warning(f"[Child] FastAPI server starting on port {port}...")
    server.run(sockets=[sock])

    sock.close()

def start_rfork_server(local_seed_key, rfork_transfer_engine_info) -> int:
    port_queue = queue.Queue()
    process = threading.Thread(
        target=start_fastapi_server,
        args=(port_queue, local_seed_key, rfork_transfer_engine_info),
        daemon=True
    )
    process.start()

    try:
        port = port_queue.get(timeout=15)
        if port == -1:
            raise RuntimeError("Child process failed to start server")
    except Exception as e:
        logger.error(f"Error: {e}")
        return -1

    start_time = time.time()
    timeout_seconds = 30
    while True:
        time.sleep(0.01)
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            logger.error(
                f"[Parent] Health check timeout after {timeout_seconds}s for port {port}. "
                "Server failed to start."
            )
            return -1
        url = f"http://127.0.0.1:{port}/health_check_with_key"
        try:
            response = requests.get(url, params={"seed_key": local_seed_key}, timeout=10)
            logger.warning(f"[Parent] GET {url} -> {response.status_code}")
            if response.status_code == 200:
                break
        except Exception as e:
            logger.warning(f"[Parent] GET {url} got error: {e}. Retry")
    return port
