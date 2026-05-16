import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .base import LLMBackend
from config import DEVICE

class LocalCausalLMBackend(LLMBackend):
    def __init__(self, name: str, model_id: str, capabilities: dict):
        super().__init__()
        self._name = name
        self.model_id = model_id
        self._capabilities = capabilities
        self.device = DEVICE

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> dict:
        return self._capabilities

    def is_available(self) -> bool:
        try:
            import transformers
            import accelerate
            import bitsandbytes
            return True
        except ImportError:
            return False

    def load(self):
        if self.model is not None:
            return
            
        print(f"Loading Translation LLM ({self.name} - {self.model_id})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        if self.device == "cuda":
            # 4-bit quantization: ~4.5GB GPU instead of ~18GB, keeps everything on GPU (faster than CPU offload)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="cuda",
                attn_implementation="sdpa",
            )
            # Re-enable TF32 (pyannote disables it during transcription)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
                attn_implementation="sdpa",
            )

    def unload(self):
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate(self, messages: list, max_new_tokens: int = 4096, multiline: bool = False, **kwargs) -> str:
        if self.model is None:
            self.load()
            
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # CRITICAL: disable Qwen3/Reasoning models internal reasoning
        )
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)

        # Default generation args
        # Set pad_token_id explicitly to suppress transformers warning
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": kwargs.get("do_sample", False),
            "temperature": kwargs.get("temperature", 0.0),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0)
        }

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                **gen_kwargs
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def generate_batch(self, messages_list: list, max_new_tokens_list: list, **kwargs) -> list:
        """Batched inference using left-padding. Processes all prompts in one GPU call."""
        if self.model is None:
            self.load()

        # Left-padding is mandatory for decoder-only LLMs in batch mode
        prev_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Silence the "Setting pad_token_id to eos_token_id" warning from model.generate()
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        texts = [
            self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for msgs in messages_list
        ]

        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        padded_len = model_inputs.input_ids.shape[1]

        gen_kwargs = {
            "max_new_tokens": max(max_new_tokens_list),
            "do_sample": kwargs.get("do_sample", False),
            "temperature": kwargs.get("temperature", 0.0),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
        }

        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, **gen_kwargs)

        # Extract only new tokens (after the left-padded prompt length)
        new_ids = [generated_ids[i][padded_len:] for i in range(len(texts))]
        responses = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)

        self.tokenizer.padding_side = prev_padding_side
        return responses
