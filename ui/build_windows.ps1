[CmdletBinding()]
param([string]$Python = ".\.venv\Scripts\python.exe")
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    $ffmpeg = (Get-Command ffmpeg.exe -CommandType Application | Select-Object -First 1).Source
    $ffprobe = (Get-Command ffprobe.exe -CommandType Application | Select-Object -First 1).Source
    & $Python -m PyInstaller --noconfirm --clean --windowed --onedir --name VoiceProtect --add-data "assets;assets" --add-data "styles.qss;." --add-binary "$ffmpeg;." --add-binary "$ffprobe;." main.py
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed ($LASTEXITCODE)" }
} finally {
    Pop-Location
}
