module.exports = {
  run: [
    // 1. Remove all files and subfolders inside output directory
    {
      method: "shell.run",
      params: {
        message: "powershell -Command \"if (Test-Path 'output') { Get-ChildItem -Path 'output' -Recurse | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue; Write-Host 'All exported files in output/ removed.' } else { New-Item -ItemType Directory -Path 'output' }\""
      }
    },
    // 2. Ensure empty output directory exists
    {
      method: "shell.run",
      params: {
        message: "powershell -Command \"if (-not (Test-Path 'output')) { New-Item -ItemType Directory -Path 'output' }\""
      }
    },
    // 3. Notify user
    {
      method: "notify",
      params: {
        title: "Output Cleared",
        description: "All exported videos, audios, and archives in 'output/' have been cleaned successfully!"
      }
    }
  ]
}
