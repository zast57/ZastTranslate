module.exports = {
  run: [
    // 1. Remove temporary working files
    {
      method: "fs.rm",
      params: {
        path: "temp"
      }
    },
    // 2. Recreate empty temp directory
    {
      method: "shell.run",
      params: {
        message: "powershell -Command \"if (-not (Test-Path 'temp')) { New-Item -ItemType Directory -Path 'temp' }\""
      }
    },
    // 3. Notify user
    {
      method: "notify",
      params: {
        title: "Cache Cleared",
        description: "Temporary workspace files in 'temp/' have been cleaned successfully!"
      }
    }
  ]
}
