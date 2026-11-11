param([ValidateSet(1,2)][int]$Workers=2)
$ErrorActionPreference='Stop'
$MultimapTaskRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $MultimapTaskRoot
& 'D:\softwares\Anaconda3\envs\astar-drl\python.exe' -X utf8 -u scripts/multimap_layout_supervisor.py --run --workers $Workers
exit $LASTEXITCODE
