import os
import gc
import torch
from .base import LLMBackend
from .local_causal import LocalCausalLMBackend
from .llama_cpp_backend import LlamaCppBackend

class AcceleratedLLMBackend(LLMBackend):
    """
    Unified LLM backend that automatically uses the high-speed C++ engine (llama-cpp / GGUF)
    when available, with transparent fallback to PyTorch NF4 (transformers).
    The user only sees the clean model name without technical jargon.
    """
    def __init__(self, name: str, model_id: str, gguf_repo: str = None, gguf_file: str = None, capabilities: dict = None):
        super().__init__()
        self._name = name
        self.model_id = model_id
        self.gguf_repo = gguf_repo
        self.gguf_file = gguf_file
        self._capabilities = capabilities or {"languages": "all", "vram_gb": 5.0}
        self.active_engine = None
        self._backend = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> dict:
        return self._capabilities

    def is_available(self) -> bool:
        # Available if either llama_cpp or transformers is installed
        cpp_ok = LlamaCppBackend(self.name, self.gguf_repo or "", self.gguf_file or "", {}).is_available() if self.gguf_repo else False
        pt_ok = LocalCausalLMBackend(self.name, self.model_id, {}).is_available()
        return cpp_ok or pt_ok

    def load(self):
        if self._backend is not None:
            return

        # Check if local GGUF already exists
        gguf_dir = os.path.expanduser("~/.zasttranslate/models/gguf")
        gguf_local_path = os.path.join(gguf_dir, self.gguf_file) if self.gguf_file else None
        has_local_gguf = gguf_local_path and os.path.exists(gguf_local_path)

        # Check if local PyTorch model already exists
        hf_dir = os.path.expanduser("~/.zasttranslate/models/hub")
        pt_folder = f"models--{self.model_id.replace('/', '--')}"
        has_local_pt = os.path.exists(os.path.join(hf_dir, pt_folder))

        # Decision: Use C++ GGUF if llama_cpp is available and (local GGUF exists OR local PyTorch doesn't exist)
        cpp_backend = LlamaCppBackend(self.name, self.gguf_repo, self.gguf_file, self._capabilities) if self.gguf_repo else None
        use_cpp = cpp_backend and cpp_backend.is_available() and (has_local_gguf or not has_local_pt)

        if use_cpp:
            try:
                print(f"[{self.name}] Initializing High-Speed C++ Engine (GGUF / llama-cpp + KV Cache)...")
                cpp_backend.load()
                self._backend = cpp_backend
                self.active_engine = "cpp"
                print(f"[{self.name}] C++ Engine active and ready.")
                return
            except Exception as e:
                print(f"[{self.name}] C++ engine load notice: {e}. Falling back to PyTorch NF4...")

        # Fallback / Default: PyTorch NF4 engine
        print(f"[{self.name}] Initializing PyTorch NF4 Engine (transformers)...")
        pt_backend = LocalCausalLMBackend(self.name, self.model_id, self._capabilities)
        pt_backend.load()
        self._backend = pt_backend
        self.active_engine = "pytorch"
        print(f"[{self.name}] PyTorch NF4 Engine active and ready.")

    def unload(self):
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
            self.active_engine = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate(self, messages: list, max_new_tokens: int = 4096, multiline: bool = False, **kwargs) -> str:
        if self._backend is None:
            self.load()
        return self._backend.generate(messages, max_new_tokens=max_new_tokens, multiline=multiline, **kwargs)

    def generate_batch(self, messages_list: list, max_new_tokens_list: list, **kwargs) -> list:
        if self._backend is None:
            self.load()
        return self._backend.generate_batch(messages_list, max_new_tokens_list, **kwargs)
