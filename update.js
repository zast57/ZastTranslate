module.exports = {
    run: [
        // 1. Pull latest code
        {
            method: "shell.run",
            params: {
                message: "git pull --rebase --autostash"
            }
        },
        // 1.5 Write temporary overrides file to resolve whisperx conflicts during update
        {
            method: "fs.write",
            params: {
                path: "overrides.txt",
                text: "huggingface-hub>=0.25.0\ntransformers==4.57.3"
            }
        },
        // 2. Update Python dependencies
        {
            method: "shell.run",
            params: {
                venv: "env",
                path: ".",
                env: {
                    "UV_NATIVE_TLS": "true",
                    "UV_INSECURE_HOST": "pypi.org,pypi.python.org,files.pythonhosted.org",
                    "PIP_TRUSTED_HOST": "pypi.org pypi.python.org files.pythonhosted.org"
                },
                message: [
                    "uv pip install -r requirements.txt --upgrade --override overrides.txt"
                ]
            }
        },
        // 3. Force-update yt-dlp to latest (YouTube API changes frequently)
        {
            method: "shell.run",
            params: {
                venv: "env",
                path: ".",
                env: {
                    "UV_NATIVE_TLS": "true",
                    "UV_INSECURE_HOST": "pypi.org,pypi.python.org,files.pythonhosted.org",
                    "PIP_TRUSTED_HOST": "pypi.org pypi.python.org files.pythonhosted.org"
                },
                message: ["uv pip install -U yt-dlp"]
            }
        },
        // 4. Ensure correct PyTorch version
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
        // 4.5 Clean up overrides file
        {
            method: "fs.rm",
            params: {
                path: "overrides.txt"
            }
        },
        // 5. Done
        {
            method: "notify",
            params: {
                title: "Update Complete",
                description: "ZastTranslate has been updated to the latest version."
            }
        }
    ]
}
