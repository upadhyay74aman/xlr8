import sys
import subprocess
import shutil
import os

def get_vram_info() -> dict:
    """
    Detects the operating system and hardware architecture to calculate
    the total and available VRAM in Megabytes (MB). Highly optimized for Windows.
    """
    vram_data = {"platform": "unknown", "total_mb": 0, "available_mb": 0}
    
    # 1. Check for Apple Silicon (Mac M1/M2/M3/M4)
    if sys.platform == "darwin":
        try:
            cmd = "sysctl -n hw.optional.arm64"
            is_arm64 = subprocess.check_output(cmd, shell=True).decode().strip()
            if is_arm64 == "1":
                vram_data["platform"] = "apple_silicon"
                mem_cmd = "sysctl -n hw.memsize"
                total_bytes = int(subprocess.check_output(mem_cmd, shell=True).decode().strip())
                total_mb = total_bytes // (1024 * 1024)
                vram_data["total_mb"] = total_mb
                vram_data["available_mb"] = int(total_mb * 0.7)
                return vram_data
        except Exception:
            pass

    # 2. Check for NVIDIA Windows/Linux systems
    # On Windows, nvidia-smi might not be in the PATH variable by default, so we look for it.
    nvidia_smi_path = shutil.which("nvidia-smi")
    if not nvidia_smi_path and sys.platform == "win32":
        # Standard Windows installation path for NVIDIA drivers
        default_path = r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
        if os.path.exists(default_path):
            nvidia_smi_path = default_path

    if nvidia_smi_path is not None:
        try:
            cmd = f'"{nvidia_smi_path}" --query-gpu=memory.total,memory.free --format=csv,nounits,noheader'
            output = subprocess.check_output(cmd, shell=True).decode().strip()
            total, free = map(int, output.split(","))
            
            vram_data["platform"] = "nvidia"
            vram_data["total_mb"] = total
            vram_data["available_mb"] = free
            return vram_data
        except Exception:
            pass

    # 3. Fallback to CPU-only execution if no specialized GPU is found
    vram_data["platform"] = "cpu_only"
    return vram_data

def calculate_layer_budget(model_size_gb: float, total_layers: int = 32) -> int:
    """
    Calculates how many layers of a model can be offloaded to the GPU
    based on current available VRAM, keeping a safety buffer.
    """
    hardware = get_vram_info()
    available_vram = hardware["available_mb"]
    
    # Convert model size to MB and add a 512MB baseline safety buffer for the context window
    model_size_mb = model_size_gb * 1024
    safety_buffer_mb = 512
    
    effective_vram = available_vram - safety_buffer_mb
    
    if effective_vram <= 0:
        return 0  # Not enough VRAM even for basic operations, run entirely on CPU
        
    if effective_vram >= model_size_mb:
        return total_layers  # The entire model fits beautifully into VRAM
        
    # Calculate approximate memory cost per single layer
    mb_per_layer = model_size_mb / total_layers
    max_layers_allowed = int(effective_vram / mb_per_layer)
    
    return min(max_layers_allowed, total_layers)

if __name__ == "__main__":
    print("--- Xlr8 Hardware Diagnostic ---")
    info = get_vram_info()
    print(f"Detected Platform: {info['platform'].upper()}")
    print(f"Total Memory Pool: {info['total_mb']} MB")
    print(f"Available/Safe Memory: {info['available_mb']} MB")
    
    # Simulate a standard quantized 7B model (~4.5 GB file size, 32 layers)
    recommended_layers = calculate_layer_budget(model_size_gb=4.5, total_layers=32)
    print(f"\nFor a 4.5GB 7B Model, Xlr8 recommends offloading: {recommended_layers}/32 layers to GPU.")