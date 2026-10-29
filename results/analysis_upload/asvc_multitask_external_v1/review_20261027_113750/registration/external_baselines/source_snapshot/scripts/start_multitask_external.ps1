param(
    [ValidateSet('Prepare','Start','Resume','Status','Report','Pairs')][string]$Action = 'Status',
    [string]$Python = 'D:\softwares\Anaconda3\envs\astar-drl\python.exe',
    [ValidateSet(1,2)][int]$Workers = 2
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Output = Join-Path $Root 'outputs/asvc_multitask_external_v1'
if ($Action -eq 'Prepare') {
    & $Python -X utf8 -u scripts/run_multitask_external.py prepare
} elseif ($Action -eq 'Status') {
    & $Python -X utf8 scripts/run_multitask_external.py status
    if (Test-Path -LiteralPath $Output) { Get-ChildItem -LiteralPath $Output -Filter status.json -Recurse | ForEach-Object { Write-Output $_.FullName; Get-Content -LiteralPath $_.FullName -Encoding UTF8 } }
} elseif ($Action -eq 'Report') {
    & $Python -X utf8 scripts/analyze_multitask_external.py
} elseif ($Action -eq 'Pairs') {
    & $Python -X utf8 scripts/diagnose_multitask_pairs.py
} else {
    if (-not (Test-Path -LiteralPath 'data/asvc_multitask_external_v1/frozen_protocol.json')) { throw 'Prepare and pass checks first.' }
    if ($Action -eq 'Start' -and (Test-Path -LiteralPath (Join-Path $Output 'pipeline_status.json'))) { throw 'Existing coordinator record; inspect and use Resume.' }
    New-Item -ItemType Directory -Path $Output -Force | Out-Null
    $Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $Arguments = @('-X','utf8','-u','scripts/run_multitask_external.py','pipeline','--workers',"$Workers")
    if ($Action -eq 'Resume') { $Arguments += '--resume' }
    $Stdout = Join-Path $Output "pipeline_$Stamp.log"
    $Stderr = Join-Path $Output "pipeline_$Stamp.err.log"
    $Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root -WindowStyle Hidden -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    [pscustomobject]@{Pid=$Process.Id;Started=(Get-Date).ToString('o');Stdout=$Stdout;Stderr=$Stderr} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Output "launch_$Stamp.json") -Encoding UTF8
    Write-Output "External baseline coordinator PID $($Process.Id); $Stdout"
}
if ($LASTEXITCODE -ne 0) { throw 'External command failed; inspect retained logs.' }
