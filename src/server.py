import json
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI(title="Xlr8 Core Proxy Engine")

# The internal port where our speculative inference.py process lives
INTERNAL_ENGINE_URL = "http://127.0.0.1:8001"

@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    """
    Intercepts standard OpenAI-compatible requests from IDEs like Cursor,
    pipes them to our background speculative engine, and streams tokens back.
    """
    body = await request.json()
    
    # Extract incoming message data payloads
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.7)
    stream_requested = body.get("stream", True)

    # Re-structure the payload for the underlying llama-server structure
    engine_payload = {
        "prompt": format_messages_to_prompt(messages),
        "temperature": temperature,
        "stream": stream_requested
    }

    # Use an async HTTP client to stream tokens from our internal backend engine
    async def token_stream_generator():
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST", 
                f"{INTERNAL_ENGINE_URL}/completion", 
                json=engine_payload
            ) as response:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    # Read the underlying raw engine output chunk
                    raw_data = line.replace("data: ", "").strip()
                    try:
                        parsed = json.loads(raw_data)
                        content_token = parsed.get("content", "")
                        
                        # Translate it cleanly into standard OpenAI format for Cursor
                        openai_chunk = {
                            "choices": [{
                                "delta": {"content": content_token},
                                "finish_reason": None if not parsed.get("stop", False) else "stop"
                            }]
                        }
                        yield f"data: {json.dumps(openai_chunk)}\n\n"
                    except Exception:
                        pass
                yield "data: [DONE]\n\n"

    return StreamingResponse(token_stream_generator(), media_type="text/event-stream")

def format_messages_to_prompt(messages: list) -> str:
    """
    Converts structured conversation lists into raw text loops 
    that the GGUF models process cleanly.
    """
    formatted = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted

if __name__ == "__main__":
    import uvicorn
    print("--- Starting Xlr8 Core API Proxy ---")
    uvicorn.run(app, host="127.0.0.1", port=8000)