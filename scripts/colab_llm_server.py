# ============================================================
# COLAB LLM SERVER - Copy-paste this into Google Colab
# ============================================================
# 
# This notebook hosts Qwen 2.5 model and exposes it via ngrok.
# After running, copy the ngrok URL to your local environment.
#
# Usage:
#   1. Run all cells in Colab
#   2. Copy the ngrok URL (looks like https://xxxx.ngrok.io/v1/completions)
#   3. Set it locally: set LLM_ENDPOINT=https://xxxx.ngrok.io/v1/completions
#   4. Run your API server locally
#
# ============================================================

# CELL 1: Install dependencies
# ----------------------------
"""
!pip install -q transformers accelerate torch pyngrok flask
"""

# CELL 2: Authentication (run separately)
# ----------------------------------------
"""
# Get your free ngrok authtoken from: https://dashboard.ngrok.com/get-started/your-authtoken
from google.colab import userdata
NGROK_TOKEN = userdata.get('NGROK_TOKEN')  # Or manually paste: "your_token_here"

from pyngrok import ngrok
ngrok.set_auth_token(NGROK_TOKEN)
"""

# CELL 3: Load the model
# ----------------------
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Choose model based on GPU availability
# - T4 GPU (free Colab): Use 3B
# - No GPU: Use 0.5B
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"  # Or "Qwen/Qwen2.5-0.5B-Instruct" for CPU

print(f"Loading {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True
)

print(f"Model loaded! Device: {model.device}")
"""

# CELL 4: Create Flask server
# ---------------------------
"""
from flask import Flask, request, jsonify
import threading

app = Flask(__name__)

@app.route('/v1/completions', methods=['POST'])
def completions():
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 512)
    temperature = data.get('temperature', 0)
    stop = data.get('stop', [])
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with deterministic settings
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=float(temperature) if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only new tokens
    generated = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    # Handle stop tokens
    for stop_token in stop:
        if stop_token in generated:
            generated = generated.split(stop_token)[0]
    
    return jsonify({
        "choices": [{
            "text": generated,
            "finish_reason": "stop"
        }],
        "model": MODEL_ID
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": MODEL_ID})

# Run in background thread
def run_server():
    app.run(host='0.0.0.0', port=5000, threaded=True)

thread = threading.Thread(target=run_server, daemon=True)
thread.start()
print("Flask server started on port 5000")
"""

# CELL 5: Expose via ngrok
# ------------------------
"""
from pyngrok import ngrok

# Open tunnel
tunnel = ngrok.connect(5000)
public_url = tunnel.public_url  # <-- Extract the actual URL string
print("=" * 60)
print("🚀 LLM SERVER READY!")
print("=" * 60)
print(f"\\nPublic URL: {public_url}")
print(f"\\nEndpoint:   {public_url}/v1/completions")
print(f"\\n👆 Use this in your local environment:")
print(f'\\n   set LLM_ENDPOINT={public_url}/v1/completions')
print(f"\\n   # Or on Linux/Mac:")
print(f'   export LLM_ENDPOINT={public_url}/v1/completions')
print("=" * 60)
"""

# CELL 6: Test the server (optional)
# ----------------------------------
"""
import requests

test_prompt = '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Say "Hello, I am working!" and nothing else.<|im_end|>
<|im_start|>assistant
'''

response = requests.post(
    f"{public_url}/v1/completions",
    json={
        "prompt": test_prompt,
        "max_tokens": 50,
        "temperature": 0
    }
)

print("Test response:", response.json())
"""

# ============================================================
# FULL SINGLE-CELL VERSION (Copy this entire block)
# ============================================================
FULL_NOTEBOOK = '''
# CELL 1: Setup and Install
!pip install -q transformers accelerate torch pyngrok flask

# CELL 2: Configure ngrok (get token from https://dashboard.ngrok.com)
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")  # <-- PASTE YOUR TOKEN

# CELL 3: Load Model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
print(f"Loading {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
print(f"Model loaded on {model.device}")

# CELL 4: Flask Server + ngrok
from flask import Flask, request, jsonify
import threading

app = Flask(__name__)

@app.route('/v1/completions', methods=['POST'])
def completions():
    data = request.json
    inputs = tokenizer(data['prompt'], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=data.get('max_tokens', 512),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    for stop in data.get('stop', []):
        if stop in generated:
            generated = generated.split(stop)[0]
    return jsonify({"choices": [{"text": generated}], "model": MODEL_ID})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": MODEL_ID})

threading.Thread(target=lambda: app.run(port=5000), daemon=True).start()
tunnel = ngrok.connect(5000)
public_url = tunnel.public_url  # Extract the URL string
print(f"\\n{'='*60}")
print(f"🚀 READY! Set this in your terminal:")
print(f"   set LLM_ENDPOINT={public_url}/v1/completions")
print(f"{'='*60}")
'''

print(FULL_NOTEBOOK)
