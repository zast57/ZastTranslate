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
                    "pip install -r requirements.txt --upgrade"
                ]
            }
        },
        // 3. Ensure correct PyTorch version
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
        // 4. Done
        {
            method: "notify",
            params: {
                title: "Update Complete",
                description: "ZastTranslate has been updated to the latest version."
            }
        }
    ]
}
