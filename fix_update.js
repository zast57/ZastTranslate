module.exports = {
    run: [
        {
            method: "shell.run",
            params: {
                message: "git pull --rebase --autostash"
            }
        },
        {
            method: "notify",
            params: {
                html: "<b>Update mechanism fixed!</b><br>You can now safely click the normal 'Upgrade' button."
            }
        }
    ]
}
