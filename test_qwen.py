import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers import AutoConfig

def _compute_default_rope_parameters(config, device=None, **kwargs):
    base = getattr(config, "rope_theta", 10000.0)
    dim = getattr(config, "head_dim", getattr(config, "hidden_size", 1024) // getattr(config, "num_attention_heads", 16))
    import torch
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32) / dim))
    return inv_freq, 1.0

if "default" not in ROPE_INIT_FUNCTIONS:
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerConfig
if not hasattr(Qwen3TTSTalkerConfig, 'pad_token_id'):
    Qwen3TTSTalkerConfig.pad_token_id = None

from qwen_tts import Qwen3TTSModel
model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
model = Qwen3TTSModel.from_pretrained(
    model_id,
    device_map="cuda",
    attn_implementation="sdpa",
)

kwargs = {
    "text": "Hello, this is a test.",
    "language": "English",
    "x_vector_only_mode": True,
    "ref_audio": "test.wav", # we need a dummy wav
}

# create dummy wav
import soundfile as sf
import numpy as np
sf.write("test.wav", np.zeros(24000*3), 24000)

wavs, sr = model.generate_voice_clone(**kwargs)
print("Success!")
