param(
    [ValidateSet('Prepare','Start','Resume','Status','Report','Diagnose')][string]$Action='Status',
    [string]$Python='D:\softwares\Anaconda3\envs\astar-drl\python.exe',
    [ValidateSet(1,2)][int]$Workers=1
)
$ErrorActionPreference='Stop'
$global:LASTEXITCODE=0
$Root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Output=Join-Path $Root 'outputs/asvc_multitask_td_control_v1'
if ($Action -eq 'Prepare') { & $Python -X utf8 -u scripts/run_td_control.py prepare }
elseif ($Action -eq 'Status') {
    & $Python -X utf8 scripts/run_td_control.py status
    if (Test-Path -LiteralPath $Output) { Get-ChildItem -LiteralPath $Output -Filter status.json -Recurse | ForEach-Object { Write-Output $_.FullName;Get-Content -LiteralPath $_.FullName -Encoding UTF8 } }
}
elseif ($Action -eq 'Report') { & $Python -X utf8 scripts/analyze_td_control.py }
elseif ($Action -eq 'Diagnose') { & $Python -X utf8 -u scripts/diagnose_multitask_loops.py }
else {
    if (-not (Test-Path -LiteralPath 'data/asvc_multitask_td_control_v1/frozen_protocol.json')) { throw 'Prepare and pass checks first.' }
    if ($Action -eq 'Start' -and (Test-Path -LiteralPath (Join-Path $Output 'pipeline_status.json'))) { throw 'Existing coordinator record; inspect before Resume.' }
    New-Item -ItemType Directory -Path $Output -Force | Out-Null
    $Stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
    $Arguments=@('-X','utf8','-u','scripts/run_td_control.py','pipeline','--workers',"$Workers")
    if ($Action -eq 'Resume') { $Arguments+='--resume' }
    $Process=Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $Output "pipeline_$Stamp.log") -RedirectStandardError (Join-Path $Output "pipeline_$Stamp.err.log")
    [pscustomobject]@{Pid=$Process.Id;Started=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Output "launch_$Stamp.json") -Encoding UTF8
    Write-Output "TD-control coordinator PID $($Process.Id); only seed0 and seed1."
}
if ($LASTEXITCODE -ne 0) { throw 'TD-control command failed; inspect retained logs.' }
