module.exports = {
  run: [
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
          "uv pip install diffusers accelerate sentencepiece protobuf"
        ]
      }
    },
    {
      method: "notify",
      params: {
        title: "FLUX Installation Complete",
        description: "FLUX.1-schnell dependencies installed successfully! You can now generate fast 4K thumbnails in Tab 7."
      }
    }
  ]
}
