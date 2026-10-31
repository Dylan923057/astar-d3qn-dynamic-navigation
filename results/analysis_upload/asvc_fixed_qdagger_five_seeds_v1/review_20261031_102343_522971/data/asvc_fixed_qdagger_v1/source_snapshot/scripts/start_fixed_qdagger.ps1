param([switch]$Resume)
$ErrorActionPreference='Stop'
$projectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonExecutable='D:\softwares\Anaconda3\envs\astar-drl\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) { $pythonExecutable=(Get-Command python -ErrorAction Stop).Source }
Push-Location -LiteralPath $projectRoot
try {
    if (Test-Path -LiteralPath 'outputs/asvc_fixed_qdagger_v1/runner.lock') { throw 'QDagger lock exists; preserve and inspect existing work.' }
    $otherPython=@(Get-Process python,pythonw -ErrorAction SilentlyContinue)
    if ($otherPython.Count -gt 0) { throw ('Existing Python work preserved; not starting QDagger. PIDs: '+(($otherPython | ForEach-Object Id)-join ',')) }
    & $pythonExecutable -X utf8 -u scripts/run_fixed_qdagger.py --prepare
    if ($LASTEXITCODE -ne 0) { throw 'QDagger preflight failed; no training started.' }
    $logRoot=Join-Path $projectRoot 'outputs/asvc_fixed_qdagger_v1/queue_logs'
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
    $stamp=Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $stdout=Join-Path $logRoot ($stamp+'_stdout.log');$stderr=Join-Path $logRoot ($stamp+'_stderr.log')
    $arguments=@('-X','utf8','-u','scripts/run_fixed_qdagger.py','--train')
    if ($Resume) { $arguments+='--resume' }
    $process=Start-Process -FilePath $pythonExecutable -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    try { $process.PriorityClass='BelowNormal' } catch { Write-Warning 'New process priority could not be lowered.' }
    $launch=[ordered]@{pid=$process.Id;launched=(Get-Date -Format o);new_seeds=@(0,1);offline_updates=20000;online_steps=200000;sequential=$true;stdout=$stdout;stderr=$stderr;resume=[bool]$Resume}
    [IO.File]::WriteAllText((Join-Path $projectRoot 'results/asvc_fixed_qdagger_v1/launch.json'),($launch|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
    $launch|ConvertTo-Json
}
finally { Pop-Location }
