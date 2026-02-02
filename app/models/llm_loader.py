"""
LLM Loader Module
-----------------
Provides a unified interface for LLM inference with multiple backends:
- Colab + ngrok (development)
- HuggingFace Transformers (local/HF Spaces deployment)

Architecture Note:
    The LLMBackend ABC ensures seamless switching between backends.
    For deployment on HF Spaces, switch to TransformersBackend.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Optional
import requests

from app.core.logger import get_logger

logger = get_logger(__name__)


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier for logging."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text from prompt with temperature=0 (deterministic)."""
        pass


class ColabNgrokBackend(LLMBackend):
    """
    Backend for Colab-hosted LLM via ngrok tunnel.
    
    Expects an OpenAI-compatible /v1/completions endpoint.
    Set LLM_ENDPOINT environment variable to the ngrok URL.
    """
    
    def __init__(self, endpoint_url: Optional[str] = None):
        self.endpoint_url = endpoint_url or os.getenv("LLM_ENDPOINT")
        if not self.endpoint_url:
            raise ValueError(
                "LLM_ENDPOINT not set. Set it to your Colab ngrok URL, e.g., "
                "'https://xxxx.ngrok.io/v1/completions'"
            )
        self._model_name = os.getenv("LLM_MODEL_NAME", "qwen2.5-colab")
        logger.info(f"ColabNgrokBackend initialized with endpoint: {self.endpoint_url}")
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Call the Colab-hosted LLM.
        Expects OpenAI-compatible API format.
        """
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,  # CRITICAL: Determinism enforced
                "stop": ["</answer>", "\n\nQuestion:"]  # Stop tokens
            }
            
            response = requests.post(
                self.endpoint_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120  # 2 min timeout for slow inference
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Handle different response formats
            if "choices" in result:
                # OpenAI format
                return result["choices"][0].get("text", "").strip()
            elif "generated_text" in result:
                # HuggingFace TGI format
                return result["generated_text"].strip()
            elif "response" in result:
                # Custom format
                return result["response"].strip()
            else:
                logger.error(f"Unexpected response format: {result}")
                return ""
                
        except requests.exceptions.Timeout:
            logger.error("LLM request timed out")
            raise RuntimeError("LLM inference timed out. Try a smaller model or check Colab.")
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM request failed: {e}")
            raise RuntimeError(f"Failed to connect to LLM endpoint: {e}")


class TransformersBackend(LLMBackend):
    """
    Backend using HuggingFace Transformers locally.
    
    Best for: HF Spaces deployment, offline/air-gapped environments.
    Uses Qwen 2.5 0.5B by default (CPU-compatible).
    """
    
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self._model_name = model_id
        self._model = None
        self._tokenizer = None
        self._loaded = False
        logger.info(f"TransformersBackend configured for: {model_id} (lazy load)")
    
    def _load_model(self):
        """Lazy load model on first inference."""
        if self._loaded:
            return
            
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading model: {self._model_name}...")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=torch.float32,  # CPU compatible
                device_map="auto",
                low_cpu_mem_usage=True
            )
            self._loaded = True
            logger.info("Model loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate using local transformers model."""
        self._load_model()
        
        import torch
        
        inputs = self._tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.0,  # CRITICAL: Determinism
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        generated = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        return generated.strip()


# ============================================================
# Factory Function
# ============================================================

_cached_backend: Optional[LLMBackend] = None

def get_llm_backend() -> LLMBackend:
    """
    Factory function to get the appropriate LLM backend.
    
    Selection logic:
    1. If LLM_ENDPOINT is set → ColabNgrokBackend (dev mode)
    2. Otherwise → TransformersBackend (local/deployment mode)
    
    The backend is cached for reuse.
    """
    global _cached_backend
    
    if _cached_backend is not None:
        return _cached_backend
    
    if os.getenv("LLM_ENDPOINT"):
        logger.info("Using ColabNgrokBackend (LLM_ENDPOINT detected)")
        _cached_backend = ColabNgrokBackend()
    else:
        logger.info("Using TransformersBackend (local mode)")
        _cached_backend = TransformersBackend()
    
    return _cached_backend
