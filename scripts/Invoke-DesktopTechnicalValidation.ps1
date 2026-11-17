param(
    [string]$Distribution = 'Ubuntu-22.04',
    [string]$WslPython = '/home/safespeech/.venvs/safespeech/bin/python',
    [string]$Assets = '/home/safespeech/models/melo-spec-pinned-20261118'
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$taskRoot = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $taskRoot ('runs/desktop_technical_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $logRoot -ErrorAction Stop | Out-Null

function Invoke-WslPython {
    param([string]$LogName, [string[]]$PythonArguments)
    $output = & wsl -d $Distribution --cd $taskRoot -- $WslPython @PythonArguments
    $code = $LASTEXITCODE
    $output | Out-File -Encoding utf8 (Join-Path $logRoot $LogName)
    if ($code -ne 0) { throw "WSL Python failed ($code), see $LogName" }
    return $output
}

Invoke-WslPython 'environment.json' @('scripts/desktop_environment_audit.py') | Out-Host
$smoke = Invoke-WslPython 'spec.log' @('scripts/run_desktop_technical_smoke.py', '--assets', $Assets)
$result = ($smoke | Select-Object -Last 1) | ConvertFrom-Json
$batch = Invoke-WslPython 'robustness.log' @('scripts/export_spec_robustness_inputs.py', '--spec-run', $result.run, '--run-batch')
$batchResult = ($batch | Select-Object -Last 1) | ConvertFrom-Json
[pscustomobject]@{
    Scope = 'Mathematical fixture only; no human speech or finetuning'
    SpecRun = $result.run
    RobustnessSummary = $batchResult.summary
    Logs = $logRoot
} | ConvertTo-Json | Tee-Object -FilePath (Join-Path $logRoot 'result.json')
