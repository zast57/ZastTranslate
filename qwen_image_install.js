module.exports = {
  run: [
    {
      method: "log",
      params: {
        raw: "\r\n========================================================================\r\n⚠️  HARDWARE ADVISORY (QWEN-IMAGE-2.1)\r\n• Full model weight size: ~33 GB on disk\r\n• Minimum VRAM: 8 to 12 GB (NVIDIA GPU with CPU Offload)\r\n• Recommended RAM: 32 GB System RAM\r\n• Low-spec machines (4-6 GB VRAM / 16 GB RAM) may experience OOM / slowdown\r\n• This module is 100% optional: core app works perfectly without it.\r\n========================================================================\r\n\r\n"
      }
    },
    // 1. Install python dependencies into venv
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        env: {
          "UV_NATIVE_TLS": "true",
          "UV_SYSTEM_CERTS": "true",
          "UV_INSECURE_HOST": "pypi.org,pypi.python.org,files.pythonhosted.org",
          "PIP_TRUSTED_HOST": "pypi.org pypi.python.org files.pythonhosted.org"
        },
        message: [
          "uv pip install \"git+https://github.com/huggingface/diffusers.git\" \"git+https://github.com/huggingface/transformers.git\" accelerate sentencepiece protobuf pillow huggingface_hub pip-system-certs"
        ]
      }
    },
    // 2. Download Qwen-Image-2.1 model weights (~33 GB) with live progress bar
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "hf download Qwen/Qwen-Image-2.1 --local-dir models/qwen_image_2_1"
        ]
      }
    },
    // 3. Desktop notification
    {
      method: "notify",
      params: {
        title: "Qwen-Image-2.1 Installation Complete",
        description: "Qwen-Image-2.1 dependencies and model weights (~33 GB) installed successfully! You can now generate high-res visuals and thumbnails in Tab 8."
      }
    }
  ]
}
