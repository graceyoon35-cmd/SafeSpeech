$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$demoRoot = Join-Path $projectRoot 'runs\demo_noise_fix_20261124'
$userRoot = Split-Path $projectRoot -Parent
$backendRoot = Join-Path $userRoot 'SafeSpeech-desktop-integration-20261118'
$uiPath = Join-Path $demoRoot 'app_snapshot\ui'
$launcher = Join-Path $projectRoot 'scripts\launch_spec_demo.py'
$jobsPath = Join-Path $demoRoot 'live_app_jobs'
$qtPython = '/mnt/c/Users/' + (Split-Path $userRoot -Leaf) + '/SafeSpeech-desktop-integration-20261118/runs/app_integration_review_20261120/qt_test_venv/bin/python'

function Convert-DemoWslPath([string]$value) {
    $converted = & wsl.exe --distribution Ubuntu-22.04 --exec wslpath -u $value
    if ($LASTEXITCODE -ne 0) { throw 'WSL path conversion failed' }
    return ($converted | Select-Object -Last 1).Trim()
}

$uiLinux = Convert-DemoWslPath $uiPath
$backendLinux = Convert-DemoWslPath $backendRoot
$launcherLinux = Convert-DemoWslPath $launcher
$jobsLinux = Convert-DemoWslPath $jobsPath
# This uses the already installed WSL Qt runtime and WSLg display.
# Every click creates a new job; no old demonstration WAV is reused.
& wsl.exe --distribution Ubuntu-22.04 --exec $qtPython -B $launcherLinux --ui-snapshot $uiLinux --backend-repository $backendLinux --output-root $jobsLinux
if ($LASTEXITCODE -ne 0) { throw ('Demo application exited with code ' + $LASTEXITCODE) }
