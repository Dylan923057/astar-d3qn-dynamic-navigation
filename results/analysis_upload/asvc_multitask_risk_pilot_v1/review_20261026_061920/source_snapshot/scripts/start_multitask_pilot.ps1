param(
    [ValidateSet('Prepare','Start','Resume','Status','Report')][string]$Action = 'Status',
    [string]$Python = 'D:\softwares\Anaconda3\envs\astar-drl\python.exe',
    [ValidateSet(1,2)][int]$Workers = 2
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Output = Join-Path $Root 'outputs/asvc_multitask_risk_pilot_v1'
Set-Location -LiteralPath $Root
if ($Action -eq 'Prepare') {
    & $Python -u scripts/run_multitask_pilot.py prepare
    if ($LASTEXITCODE -ne 0) { throw 'Preflight failed; training was not started.' }
} elseif ($Action -eq 'Status') {
    & $Python scripts/run_multitask_pilot.py status
    if (Test-Path -LiteralPath $Output) {
        Get-ChildItem -LiteralPath $Output -Filter status.json -Recurse | ForEach-Object {
            Write-Output $_.FullName
            Get-Content -LiteralPath $_.FullName -Encoding UTF8
        }
    }
} elseif ($Action -eq 'Report') {
    & $Python scripts/analyze_multitask_pilot.py
    if ($LASTEXITCODE -ne 0) { throw 'Analysis failed.' }
} else {
    if (-not (Test-Path -LiteralPath 'data/asvc_multitask_risk_pilot_v1/frozen_protocol.json')) { throw 'Run Prepare first.' }
    if ($Action -eq 'Start' -and (Test-Path -LiteralPath (Join-Path $Output 'pipeline_status.json'))) { throw 'Existing pipeline record; inspect and use Resume.' }
    New-Item -ItemType Directory -Path $Output -Force | Out-Null
    $Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $Arguments = @('-u','scripts/run_multitask_pilot.py','pipeline','--workers',"$Workers")
    if ($Action -eq 'Resume') { $Arguments += '--resume' }
    $Stdout = Join-Path $Output "pipeline_$Stamp.log"
    $Stderr = Join-Path $Output "pipeline_$Stamp.err.log"
    $Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root -WindowStyle Hidden -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    [pscustomobject]@{Pid=$Process.Id;Started=(Get-Date).ToString('o');Stdout=$Stdout;Stderr=$Stderr} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Output "launch_$Stamp.json") -Encoding UTF8
    Write-Output "Started four-run pilot coordinator PID $($Process.Id). Log: $Stdout"
}
