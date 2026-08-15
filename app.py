import gradio as gr
from llama_cpp import Llama
import glob
import os
import sys

# 1. AUTO-FIND THE AI MODEL
model_files = glob.glob("models/*.gguf")
if not model_files:
    print("❌ ERROR: No .gguf model found in the 'models' folder!")
    print("📁 Please download a model (e.g., Mistral-7B) and place it in C:\\llama-cluster\\models")
    sys.exit(1)

model_path = model_files[0]  # Automatically picks the first model
print(f"✅ Loading model: {os.path.basename(model_path)}")

# 2. RPC SERVER CONFIGURATION (Must be running!)
RPC_SERVER = "127.0.0.1:50052"

try:
    # 3. CONNECT TO THE RPC CLUSTER AND LOAD MODEL ON GPU
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,              # Send ALL layers to the GPU via RPC
        rpc_servers=RPC_SERVER,
        chat_format="llama-2",
        verbose=False                 # Set to True for debugging
    )
    print(f"🚀 Connected to RPC Server at {RPC_SERVER}")
    print(f"🖥️  GPU Acceleration: ENABLED")

except Exception as e:
    print(f"❌ Failed to connect to RPC Server: {e}")
    print("⚠️  Make sure 'ggml-rpc-server.exe' is running in C:\\llama-cluster\\bin")
    sys.exit(1)

# 4. THE CHAT LOGIC
def chat_function(message, history):
    """Process a chat message through the LLM via the RPC GPU cluster."""
    # Convert Gradio history into the format llama.cpp expects
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    # Send the conversation to the RPC server (GPU)
    response = llm.create_chat_completion(messages=messages)

    # Return the AI's reply
    return response["choices"][0]["message"]["content"]

# 5. CREATE THE WEB UI
ui = gr.ChatInterface(
    fn=chat_function,
    title="🤖 LLaMA Cluster - GPU Powered AI",
    description=f"Active Model: {os.path.basename(model_path)} | Running on RPC Cluster (GPU)",
    examples=[["Hello, who are you?"], ["Write a short poem about AI."]]
)

# 6. LAUNCH THE SERVER
if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
    print("🌐 Web UI running at: http://127.0.0.1:7860")
