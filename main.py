import os
import sys
import asyncio
import httpx
import uvicorn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.hardware import get_vram_info
from src.router import setup_models, MODEL_MATRIX
from src.inference import InferenceOrchestrator
from src.server import app
from src.cli import run_terminal_chat

async def wait_for_proxy(host: str, port: int, timeout: int = 15) -> bool:
    """Polls the proxy server using a TCP socket check until it's accepting connections."""
    import socket
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect((host, port))
            sock.close()
            return True
        except Exception:
            await asyncio.sleep(0.5)
    return False

async def run_xlr8(model_key: str):
    print("⚡ Welcome to Xlr8 ⚡")
    print("==========================================")

    info = get_vram_info()
    print(f"[Xlr8 Engine] Hardware Tracked: {info['platform'].upper()}")
    print(f"[Xlr8 Engine] Free Compute VRAM Pool: {info['available_mb']} MB")

    try:
        target_path, draft_path = setup_models(model_key)
    except Exception as e:
        print(f"❌ [Xlr8 Setup Error] Failed to resolve models: {e}")
        return

    model_size_gb = MODEL_MATRIX[model_key]["target"]["size_gb"]

    # Step 1: Boot the inference engine and wait until healthy
    orchestrator = InferenceOrchestrator(port=8001)
    engine_started = await orchestrator.start_engine(
        target_path=target_path,
        draft_path=draft_path,
        target_size_gb=model_size_gb
    )

    if not engine_started:
        print("❌ [Xlr8 Engine Error] Stopping due to backend initialization failure.")
        return

    print("\n==========================================")
    print("🎉 Xlr8 is officially online and active!")
    print("==========================================\n")

    # Step 2: Start proxy server as background task
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="error")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    # Step 3: Wait for proxy to actually be ready ✅
    print("[Xlr8] Waiting for proxy to bind...")
    proxy_ready = await wait_for_proxy("127.0.0.1", 8000)
    if not proxy_ready:
        print("❌ [Xlr8] Proxy server failed to start in time.")
        server_task.cancel()
        return

    print("[Xlr8] Proxy ready. Launching terminal session...\n")

    try:
        # Step 4: Launch CLI — now guaranteed proxy is up
        await asyncio.to_thread(run_terminal_chat)

    except KeyboardInterrupt:
        print("\n[Xlr8 Engine] Interrupted by user.")
    finally:
        print("\n[Xlr8 Engine] Shutting down safely...")
        server_task.cancel()
        if hasattr(orchestrator, 'process') and orchestrator.process:
            try:
                orchestrator.process.terminate()
            except Exception:
                pass

if __name__ == "__main__":
    chosen_model = "qwen2.5-7b"
    asyncio.run(run_xlr8(chosen_model))
def run_xlr8_sync():
    asyncio.run(run_xlr8("qwen2.5-7b"))
