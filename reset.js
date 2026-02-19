module.exports = {
    run: [
        // 1. Remove virtual environment
        {
            method: "fs.rm",
            params: {
                path: "env"
            }
        },
        // 2. Remove temporary files
        {
            method: "fs.rm",
            params: {
                path: "temp"
            }
        },
        // 3. Done
        {
            method: "notify",
            params: {
                title: "Reset Complete",
                description: "The environment has been reset. Click 'Install' to set up again."
            }
        }
    ]
}
