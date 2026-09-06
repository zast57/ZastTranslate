module.exports = {
    daemon: true,
    run: [
        // Clean app-level Python bytecode cache
        {
            method: "fs.rm",
            params: {
                path: "__pycache__"
            }
        },
        {
            method: "fs.rm",
            params: {
                path: "modules/__pycache__"
            }
        },
        // Auto-update yt-dlp (YouTube changes its API frequently — keeps downloads working)
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
                message: ["uv pip install -U yt-dlp || echo yt-dlp update skipped"]
            }
        },
        // Launch app with no bytecode caching
        {
            method: "shell.run",
            params: {
                venv: "env",
                path: ".",
                env: {
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONUTF8": "1"
                },
                message: [
                    "python app.py",
                ],
                on: [{
                    "event": "/(http:\\/\\/[0-9.:]+)/",
                    "done": true
                }]
            }
        },
        {
            method: "local.set",
            params: {
                url: "{{input.event[1]}}"
            }
        },
    ]
}
