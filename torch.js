module.exports = {
    run: [
        // windows nvidia
        {
            "when": "{{platform === 'win32' && gpu === 'nvidia'}}",
            "method": "shell.run",
            "params": {
                "venv": "{{args && args.venv ? args.venv : null}}",
                "path": "{{args && args.path ? args.path : '.'}}",
                "message": [
                    "pip install typing-extensions>=4.10.0",
                    "pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps"
                ]
            },
            "next": null
        },
        // linux nvidia
        {
            "when": "{{platform === 'linux' && gpu === 'nvidia'}}",
            "method": "shell.run",
            "params": {
                "venv": "{{args && args.venv ? args.venv : null}}",
                "path": "{{args && args.path ? args.path : '.'}}",
                "message": [
                    "uv pip install typing-extensions>=4.10.0",
                    "uv pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps"
                ]
            },
            "next": null
        },
        // apple silicon mac
        {
            "when": "{{platform === 'darwin' && arch === 'arm64'}}",
            "method": "shell.run",
            "params": {
                "venv": "{{args && args.venv ? args.venv : null}}",
                "path": "{{args && args.path ? args.path : '.'}}",
                "message": [
                    "pip install typing-extensions>=4.10.0",
                    "pip install torch==2.8.0 torchaudio==2.8.0"
                ]
            },
            "next": null
        },
        // cpu fallback
        {
            "method": "shell.run",
            "params": {
                "venv": "{{args && args.venv ? args.venv : null}}",
                "path": "{{args && args.path ? args.path : '.'}}",
                "message": [
                    "pip install typing-extensions>=4.10.0",
                    "pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu"
                ]
            }
        }
    ]
}
