import os
import sys
import gc
import torch
from .base import LLMBackend
from config import DEVICE

# Configure CUDA DLL directory for Windows
if hasattr(os, "add_dll_directory"):
    try:
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib):
            os.add_dll_directory(torch_lib)
    except Exception:
        pass

class LlamaCppBackend(LLMBackend):
    def __init__(self, name: str, repo_id: str, filename: str, capabilities: dict):
        super().__init__()
        self._name = name
        self.repo_id = repo_id
        self.filename = filename
        self._capabilities = capabilities
        self.model = None
        self._cache = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> dict:
        return self._capabilities

    def is_available(self) -> bool:
        try:
            import llama_cpp
            return True
        except Exception:
            return False

    def load(self):
        if self.model is not None:
            return
            
        import llama_cpp
        print(f"Loading GGUF/C++ LLM ({self.name} - {self.repo_id}/{self.filename})...")
        
        models_dir = os.path.expanduser("~/.zasttranslate/models/gguf")
        os.makedirs(models_dir, exist_ok=True)
        
        # Calculate GPU layers: -1 for all layers on GPU if CUDA
        n_gpu = -1 if DEVICE == "cuda" else 0
        
        self.model = llama_cpp.Llama.from_pretrained(
            repo_id=self.repo_id,
            filename=self.filename,
            local_dir=models_dir,
            n_gpu_layers=n_gpu,
            n_ctx=4096,
            verbose=False,
        )
        
        # Enable Prompt Caching (KV Cache) in RAM
        # 512 MB cache stores prompt prefixes (system prompts, instructions)
        # So subsequent segments sharing the same system prompt evaluate in ~0ms
        try:
            self._cache = llama_cpp.LlamaRAMCache(capacity_bytes=512 * 1024 * 1024)
            self.model.set_cache(self._cache)
            print("LlamaCpp: RAM Prompt Caching (KV Cache) enabled (512MB capacity).")
        except Exception as e:
            print(f"LlamaCpp: Could not enable LlamaRAMCache: {e}")

    def unload(self):
        if self.model is not None:
            del self.model
            del self._cache
            self.model = None
            self._cache = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate(self, messages: list, max_new_tokens: int = 4096, multiline: bool = False, **kwargs) -> str:
        if self.model is None:
            self.load()
            
        temperature = kwargs.get("temperature", 0.3)
        do_sample = kwargs.get("do_sample", True)
        temp = temperature if do_sample else 0.0
        
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temp,
            repeat_penalty=kwargs.get("repetition_penalty", 1.05),
        )
        
        choice = response["choices"][0]["message"]["content"]
        return choice if choice is not None else ""

    def generate_batch(self, messages_list: list, max_new_tokens_list: list, **kwargs) -> list:
        """
        Batched generation using Prompt Caching.
        The system prompt prefix is evaluated only once on the first prompt,
        then served instantly from the KV Cache for all subsequent prompts.
        """
        if self.model is None:
            self.load()
            
        results = []
        for msgs, mnt in zip(messages_list, max_new_tokens_list):
            res = self.generate(msgs, max_new_tokens=mnt, **kwargs)
            results.append(res)
        return results
