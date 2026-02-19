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
        // Launch app with no bytecode caching
        {
            method: "shell.run",
            params: {
                venv: "env",
                path: ".",
                env: {
                    "PYTHONDONTWRITEBYTECODE": "1"
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
