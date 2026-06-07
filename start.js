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
                    "UV_INSECURE_HOST": "pypi.org,pypi.python.org,files.pythonhosted.org",
                    "PIP_TRUSTED_HOST": "pypi.org pypi.python.org files.pythonhosted.org"
                },
                message: ["uv pip install -U yt-dlp"]
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
                    "PYTHONUNBUFFERED": "1"
                },
                message: [
                    "python app.py",
                ],
                on: [{
                    "event": "/(http:\\/\\/\\S+)/",
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
