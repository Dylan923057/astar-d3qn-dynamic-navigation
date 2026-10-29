param(
    [ValidateSet('Prepare','Start','Resume','Status','Report','Pairs')][string]$Action = 'Status',
    [string]$Python = 'D:\softwares\Anaconda3\envs\astar-drl\python.exe',
    [ValidateSet(1,2)][int]$Workers = 2
)
$ErrorActionPreference = 'Stop'
# The already frozen launcher checks LASTEXITCODE after Start-Process, which
# does not set that variable. Initialize it in this thin non-scientific wrapper.
# The frozen launcher, training protocol, and all research sources stay intact.
$global:LASTEXITCODE = 0
& (Join-Path $PSScriptRoot 'start_multitask_external.ps1') -Action $Action -Python $Python -Workers $Workers
if ($LASTEXITCODE -ne 0) { throw 'External baseline command failed.' }
if ($Action -in @('Report','Pairs')) {
    & $Python -X utf8 (Join-Path $PSScriptRoot 'render_multitask_pairs.py')
    if ($LASTEXITCODE -ne 0) { throw 'Offline diagnostic rendering failed.' }
}
