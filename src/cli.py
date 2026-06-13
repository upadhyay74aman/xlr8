import sys
import httpx
import json
import argparse

# Targets our local front-facing FastAPI OpenAI-compatible proxy layer
ENGINE_URL = "http://127.0.0.1:8000/v1/chat/completions"

def parse_arguments():
    """
    Parses global system execution flags for deployment distribution.
    """
    parser = argparse.ArgumentParser(
        description="⚡ Xlr8: Speculative-Accelerated Ultra-Fast Local AI Workstation Engine ⚡"
    )
    subparsers = parser.add_subparsers(dest="command", help="Execution modes")
    
    # 'start' subcommand structure
    start_parser = subparsers.add_parser("start", help="Ignite the local background AI engine cluster")
    start_parser.add_argument(
        "--model", 
        default="qwen2.5-7b", 
        choices=["qwen2.5-7b", "llama3-8b"],
        help="Target model architecture template to deploy (Default: qwen2.5-7b)"
    )
    
    # 'chat' subcommand structure
    subparsers.add_parser("chat", help="Open an interactive streaming terminal session directly")

    return parser.parse_args()

def run_terminal_chat():
    """
    Your verified multi-turn streaming conversation environment.
    Connects to the operational proxy to serve tokens directly over stdout.
    """
    print("\n" + "="*50)
    print(" 🚀 Xlr8 Interactive Terminal Session Started")
    print(" 💡 Type 'exit' or 'quit' to close the session.")
    print("="*50 + "\n")

    # This array tracks the ongoing conversation context
    conversation_history = []

    while True:
        try:
            # 1. Capture User Input
            user_input = input("👤 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("\n👋 Closing terminal session. Keep accelerating!")
                break

            # Append user message to history
            conversation_history.append({"role": "user", "content": user_input})
            
            # Print the Assistant label and prepare for stream output
            print("🤖 Xlr8: ", end="", flush=True)

            # 2. Configure Payload for Streaming Response
            payload = {
                "model": "qwen2.5-7b",
                "messages": conversation_history,
                "stream": True,
                "temperature": 0.7
            }

            assistant_reply = ""

            # 3. Establish Stream connection using HTTPX
            with httpx.stream("POST", ENGINE_URL, json=payload, timeout=60.0) as response:
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    
                    if line.startswith("data: "):
                        raw_data = line[6:].strip()  # Strip out the "data: " prefix
                        
                        if raw_data == "[DONE]":
                            break
                        
                        try:
                            parsed = json.loads(raw_data)
                            delta = parsed["choices"][0]["delta"]
                            
                            if "content" in delta:
                                token = delta["content"]
                                assistant_reply += token
                                # Print token immediately without newline or buffering
                                sys.stdout.write(token)
                                sys.stdout.flush()
                        except Exception:
                            pass
            
            # Print a clean newline at the end of the streaming generation
            print("\n" + "-" * 50)
            
            # Commit the assistant's fully generated reply back to the chat history context
            conversation_history.append({"role": "assistant", "content": assistant_reply})

        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Execution Error: Could not reach engine background proxy ({e})")
            print("Ensure your main.py engine instance is active on your host system!\n")
            break

if __name__ == "__main__":
    # If a developer explicitly calls this file, fall back to executing chat context natively
    run_terminal_chat()