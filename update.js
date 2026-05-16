module.exports = {
    run: [
        // 1. Pull latest code
        {
            method: "shell.run",
            params: {
                message: "git pull --rebase --autostash"
            }
        },
        // 2. Update Python dependencies
        {
            method: "shell.run",
            params: {
                venv: "env",
                path: ".",
                message: [
                    "uv pip install -r requirements.txt --upgrade"
                ]
            }
        },
        // 3. Force-update yt-dlp to latest (YouTube API changes frequently)
        {
            method: "shell.run",
            params: {
                venv: "env",
                path: ".",
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
