param([switch]$Resume, [ValidateSet(1,2)][int]$Workers=2)
$ErrorActionPreference='Stop'
$TaskRoot=Split-Path -Parent $PSScriptRoot
$TaskPython='D:\softwares\Anaconda3\envs\astar-drl\python.exe'
$TaskArguments=@('-X','utf8','-u','scripts/run_multimap_layout.py','--run','--workers',"$Workers")
if ($Resume) { $TaskArguments+='--resume' }
Set-Location -LiteralPath $TaskRoot
& $TaskPython @TaskArguments
exit $LASTEXITCODE
