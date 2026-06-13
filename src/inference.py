import os
import sys
import asyncio
import shutil
from src.hardware import calculate_layer_budget

class InferenceOrchestrator:
    def __init__(self, port: int = 8001):
        self.port = port
        self.process = None

    def _find_llama_server_binary(self) -> str:
        binary_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        search_dirs = [
            project_root,
            os.path.join(project_root, "bin"),
            os.path.join(project_root, "llama.cpp"),
            os.path.join(project_root, "llama"),
            os.path.join(project_root, "backends"),
            os.getcwd(),
        ]
        
        for directory in search_dirs:
            candidate = os.path.join(directory, binary_name)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        
        binary_path = shutil.which(binary_name)
        if binary_path:
            return os.path.abspath(binary_path)

        # Auto download if not found anywhere
        print("⚙️  llama-server not found. Auto-downloading the right build for your system...")
        return self._auto_download_llama_server()

    def _auto_download_llama_server(self) -> str:
        import urllib.request
        import zipfile
        import json

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bin_dir = os.path.join(project_root, "bin")
        os.makedirs(bin_dir, exist_ok=True)

        binary_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"

        try:
            print("🌐 Fetching latest llama.cpp release info...")
            api_url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
            req = urllib.request.Request(api_url, headers={"User-Agent": "xlr8-installer"})
            with urllib.request.urlopen(req) as r:
                release = json.loads(r.read())

            assets = release["assets"]

            # Pick right build per platform — Vulkan first on Windows (no extra runtime needed)
            if sys.platform == "win32":
                keywords = ["win-vulkan-x64", "win-cuda-12", "win-cpu-x64"]
            elif sys.platform == "darwin":
                keywords = ["macos-arm64", "macos-x64"]
            else:
                keywords = ["ubuntu-vulkan-x64", "ubuntu-x64"]

            asset = None
            for keyword in keywords:
                asset = next((a for a in assets if keyword in a["name"]), None)
                if asset:
                    break

            if not asset:
                print("❌ Could not find a suitable llama.cpp build for your platform.")
                return None

            zip_path = os.path.join(bin_dir, "llama.zip")
            size_mb = asset["size"] // 1024 // 1024
            print(f"📥 Downloading {asset['name']} ({size_mb} MB)...")

            def show_progress(count, block_size, total_size):
                percent = min(int(count * block_size * 100 / total_size), 100)
                print(f"\r   Progress: {percent}%", end="", flush=True)

            urllib.request.urlretrieve(
                asset["browser_download_url"],
                zip_path,
                reporthook=show_progress
            )
            print()

            print("📦 Extracting...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(bin_dir)
            os.remove(zip_path)

            # Find the binary in extracted files
            for root, dirs, files in os.walk(bin_dir):
                for f in files:
                    if f == binary_name:
                        full_path = os.path.join(root, f)
                        print(f"✅ llama-server ready!")
                        return full_path

            print("❌ Binary not found after extraction.")
            return None

        except Exception as e:
            print(f"❌ Auto-download failed: {e}")
            print("👉 Manual install: https://github.com/ggerganov/llama.cpp/releases/latest")
            return None

    async def start_engine(self, target_path: str, draft_path: str, target_size_gb: float):
        llama_bin = self._find_llama_server_binary()
        if not llama_bin:
            print("\n❌ [Xlr8 Inference Error] 'llama-server' executable could not be found.")
            return False

        print("[Xlr8 Inference] Querying system capabilities for optimal layer distributions...")
        gpu_layers = calculate_layer_budget(model_size_gb=target_size_gb, total_layers=32)
        print(f"[Xlr8 Inference] Optimization Map: Offloading {gpu_layers}/32 layers to GPU.")

        help_text = ""
        try:
            import subprocess
            result = subprocess.run(
                [llama_bin, "-h"],
                capture_output=True,
                text=True,
                timeout=5,
                encoding="utf-8",
                errors="ignore"
            )
            help_text = result.stdout + result.stderr
        except Exception as e:
            print(f"⚠️ [Xlr8 Inference] Warning probing binary help: {e}")

        has_model_draft = "--model-draft" in help_text or " -md " in help_text
        has_ngld = "-ngld" in help_text or "--gpu-layers-draft" in help_text or "--n-gpu-layers-draft" in help_text
        supports_second_ngl = (not has_ngld) and ("-ngl" in help_text)
        draft_flag = "--model-draft" if has_model_draft else "--spec-draft-n-max"

        for mode in ["speculative", "fallback"]:
            if mode == "speculative":
                print(f"[Xlr8 Inference] Ignition sequence initiated. Launching with speculative decoding...")
                cmd_args = [
                    llama_bin,
                    "-m", target_path,
                    "-ngl", str(gpu_layers),
                    draft_flag, draft_path,
                    "--spec-draft-n-max", "8",
                    "--port", str(self.port),
                    "--ctx-size", "4096",
                    "--host", "127.0.0.1",
                ]
                if has_ngld:
                    cmd_args.extend(["-ngld", "99"])
                elif supports_second_ngl:
                    cmd_args.extend(["-ngl", "99"])
            else:
                print("⚠️ [Xlr8 Inference] Falling back to launching without draft model...")
                cmd_args = [
                    llama_bin,
                    "-m", target_path,
                    "-ngl", str(gpu_layers),
                    "--port", str(self.port),
                    "--ctx-size", "4096",
                    "--host", "127.0.0.1",
                ]

            try:
                if self.process:
                    try:
                        self.process.terminate()
                        await self.process.wait()
                    except Exception:
                        pass

                self.process = await asyncio.create_subprocess_exec(
                    *cmd_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                ready = await self._wait_for_ready(timeout=30)
                if ready:
                    print(f"🚀 [Xlr8 Inference] Engine successfully booted on port {self.port}!")
                    return True
                else:
                    print(f"❌ [Xlr8 Inference Error] Engine did not become ready during '{mode}' boot.")
                    if mode == "fallback":
                        return False
            except Exception as e:
                print(f"❌ [Xlr8 Inference Error] Crash during ignition ({mode}): {str(e)}")
                if mode == "fallback":
                    return False

        return False

    async def _wait_for_ready(self, timeout: int = 30) -> bool:
        import httpx
        url = f"http://127.0.0.1:{self.port}/health"
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if self.process.returncode is not None:
                stdout_bytes = await self.process.stdout.read()
                stderr_bytes = await self.process.stderr.read()
                stdout_str = stdout_bytes.decode(errors="ignore").strip()
                stderr_str = stderr_bytes.decode(errors="ignore").strip()
                print(f"❌ Engine crashed (exit code {self.process.returncode}).")
                if stdout_str:
                    print(f"--- Engine Stdout ---\n{stdout_str}")
                if stderr_str:
                    print(f"--- Engine Stderr ---\n{stderr_str}")
                return False
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.get(url, timeout=2.0)
                    if r.status_code == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(1)
        return False

    async def stop_engine(self):
        if self.process:
            print("[Xlr8 Inference] Spin-down requested. Terminating background engines...")
            try:
                self.process.terminate()
                await self.process.wait()
                print("✅ [Xlr8 Inference] Background server closed.")
            except Exception:
                pass