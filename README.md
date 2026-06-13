# Xlr8 — Your Free Local AI on Your GPU

> Run a powerful LLM **completely free**, **completely private**, on your own machine. No API keys. No subscriptions. No data leaving your PC.

```bash
pip install xlr8
xlr8
```

That's it. Xlr8 handles everything else automatically.

---

## What It Does

Xlr8 turns your GPU into a free, private AI assistant you can chat with from any terminal. Under the hood it uses **speculative decoding** — running two models simultaneously to generate responses 2-3x faster than a single model alone.

```
👤 You: explain how neural networks work
🤖 Xlr8: A neural network is a system of interconnected layers...
```

---

## How It Works

```
You type → CLI → Proxy (port 8000) → llama-server (port 8001) → Qwen 7B on your GPU → streams back
```

- **Target model** — Qwen 2.5 7B handles the actual responses
- **Draft model** — Qwen 0.5B runs ahead and guesses the next tokens
- If the guess is right, the 7B model accepts it for free → **2-3x faster output**
- **Everything runs locally** — your conversations never touch the internet

---

## Requirements

- Python 3.10+
- NVIDIA or AMD GPU with 6GB+ VRAM (runs on CPU too, just slower)
- Windows, Linux, or macOS

---

## Installation

```bash
pip install xlr8
xlr8
```

On first run, Xlr8 will automatically:
1. Download the Qwen 2.5 7B model (~4.7 GB)
2. Download the Qwen 0.5B draft model (~0.5 GB)
3. Download the right llama-server binary for your OS
4. Boot everything and drop you into a chat session

No manual setup required.

---

## Optional: Faster Downloads

Set a free Hugging Face token to avoid rate limiting:

```bash
# Get your token at huggingface.co/settings/tokens
export HF_TOKEN=hf_your_token_here   # Linux/Mac
$env:HF_TOKEN="hf_your_token_here"   # Windows PowerShell
```

---

## Supported Models

| Key | Model | Size | Best For |
|-----|-------|------|----------|
| `qwen2.5-7b` | Qwen 2.5 7B (default) | 4.7 GB | General tasks, coding |
| `llama3-8b` | Meta Llama 3 8B | 4.9 GB | Reasoning, analysis |

---

## Project Structure

```
xlr8/
├── main.py          # Entry point and orchestration
├── src/
│   ├── router.py    # Model management and downloads
│   ├── inference.py # llama-server process management
│   ├── server.py    # OpenAI-compatible proxy (port 8000)
│   ├── cli.py       # Terminal chat interface
│   └── hardware.py  # GPU detection and layer budgeting
└── models/          # Downloaded models stored here (gitignored)
```

---

## Use With Cursor / VS Code

Since Xlr8 exposes an OpenAI-compatible API on `http://127.0.0.1:8000`, you can point any OpenAI-compatible tool at it:

- **Base URL:** `http://127.0.0.1:8000/v1`
- **Model:** `qwen2.5-7b`
- **API Key:** any string (not validated)

---

## Why Xlr8?

| | Xlr8 | ChatGPT | GitHub Copilot |
|---|---|---|---|
| Cost | Free forever | $20/month | $10/month |
| Privacy | 100% local | Sends to OpenAI | Sends to Microsoft |
| Internet required | Download only | Always | Always |
| GPU required | Recommended | No | No |

---

## License

MIT — do whatever you want with it.

---

Built with [llama.cpp](https://github.com/ggerganov/llama.cpp) and [Qwen 2.5](https://huggingface.co/Qwen).
