import os

# ✅ Use EITHER fast transfer OR mirror — not both
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Requires: pip install hf_transfer

from huggingface_hub import hf_hub_download
import huggingface_hub

# ✅ Set token from environment to avoid rate limiting
HF_TOKEN = os.environ.get("HF_TOKEN", None)
if HF_TOKEN:
    huggingface_hub.login(token=HF_TOKEN, add_to_git_credential=False)
else:
    print("⚠️  Warning: No HF_TOKEN set. Downloads may be slow. Set it with: $env:HF_TOKEN='hf_...'")

MODEL_MATRIX = {
    "qwen2.5-7b": {
        "description": "Qwen 2.5 7B (Excellent for General Tasks & Coding)",
        "target": {
            "repo_id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
            "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            "size_gb": 4.7
        },
        "draft": {
            "repo_id": "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
            "filename": "Qwen2.5-0.5B-Instruct-Q8_0.gguf",
            "size_gb": 0.5
        }
    },
    "llama3-8b": {
        "description": "Meta Llama 3 8B (Highly Intelligent)",
        "target": {
            "repo_id": "bartowski/Meta-Llama-3-8B-Instruct-GGUF",
            "filename": "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
            "size_gb": 4.9
        },
        "draft": {
            "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
            "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            "size_gb": 0.7
        }
    }
}

def setup_models(model_key: str, destination_dir: str = None) -> tuple:
    if model_key not in MODEL_MATRIX:
        raise ValueError(f"Model '{model_key}' is not supported yet by Xlr8.")

    # ✅ Always resolve models relative to this file, not where xlr8 is run from
    if destination_dir is None:
        destination_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
        destination_dir = os.path.normpath(destination_dir)

    model_info = MODEL_MATRIX[model_key]
    os.makedirs(destination_dir, exist_ok=True)

    print(f"\n[Xlr8 Router] Setting up: {model_info['description']}")

    def download_if_missing(info: dict, label: str) -> str:
        path = os.path.join(destination_dir, info["filename"])
        if not os.path.exists(path):
            print(f"📥 {label} missing. Downloading {info['filename']} (~{info['size_gb']} GB)...")
            hf_hub_download(
                repo_id=info["repo_id"],
                filename=info["filename"],
                local_dir=destination_dir,
                token=HF_TOKEN,
            )
            print(f"✅ {label} downloaded to: {path}")
        else:
            print(f"✅ {label} found locally: {path}")
        return path

    target_path = download_if_missing(model_info["target"], "Target model")
    draft_path  = download_if_missing(model_info["draft"],  "Fast Assistant")

    return target_path, draft_path

if __name__ == "__main__":
    print("--- Xlr8 Routing Engine Testing ---")
    print("Available keys in matrix:")
    for key, val in MODEL_MATRIX.items():
        print(f"  - {key}: {val['description']}")