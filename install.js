module.exports = {
    run: [
        // 0. Write temporary overrides file to resolve whisperx vs transformers conflicts
        {
            method: "fs.write",
            params: {
                path: "overrides.txt",
                text: "huggingface-hub>=0.34.0,<1.0\ntransformers==4.57.3"
            }
        },
        // 1. Create venv and install initial dependencies (gradio + requirements.txt)
        // NOTE: whisperx pins torch~=2.8.0 which installs CPU-only torch from PyPI.
        // We MUST install requirements first, then override with CUDA torch in step 2.
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
                    "uv pip install gradio",
                    "uv pip install -r requirements.txt --override overrides.txt"
                ]
            }
        },
        // 2. Override torch with CUDA-enabled version (replaces CPU-only from step 1)
        {
            method: "script.start",
            params: {
                uri: "torch.js",
                params: {
                    venv: "env",
                    path: "."
                }
            }
        },
        // 3. Install Qwen3-TTS (--no-deps to avoid pulling its own torch)
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
                message: "uv pip install qwen-tts --no-deps"
            }
        },
        // 4. Install SoX binary (required by qwen-tts for audio processing)
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
                message: "conda install -y -c conda-forge sox"
            }
        },
        // 5. Install Qwen3-TTS non-torch dependencies
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
                message: "uv pip install sox soundfile safetensors huggingface_hub tokenizers"
            }
        },
        // 6. Try installing flash-attn (optional, speeds up TTS inference)
        {
            when: "{{platform !== 'win32'}}",
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
                message: "uv pip install wheel psutil ninja packaging && uv pip install flash-attn --no-build-isolation || echo FlashAttention not available - will use default attention"
            }
        },


        // 7. Verify full installation
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
                    "python check_env.py",
                    "python -c \"import torch; print('torch', torch.__version__, 'CUDA' if torch.cuda.is_available() else 'CPU')\"",
                    "python -c \"from qwen_tts import Qwen3TTSModel; print('Qwen3-TTS OK')\"",
                    "python -c \"import bitsandbytes; print('bitsandbytes', bitsandbytes.__version__, 'OK')\""
                ]
            }
        },
        // 8. Write sentinel file to indicate success
        {
            method: "fs.write",
            params: {
                path: "env/installed.sentinel",
                text: "success"
            }
        },
        // 9. Clean up overrides file
        {
            method: "fs.rm",
            params: {
                path: "overrides.txt"
            }
        },
        // 10. Done
        {
            method: "notify",
            params: {
                title: "Installation Complete!",
                description: "ZastTranslate is ready with voice cloning support. Click 'Start' to launch."
            }
        }
    ]
}
