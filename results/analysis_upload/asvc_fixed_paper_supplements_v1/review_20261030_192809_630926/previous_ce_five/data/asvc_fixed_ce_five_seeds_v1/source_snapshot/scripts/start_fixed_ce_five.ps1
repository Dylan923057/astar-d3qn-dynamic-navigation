param([switch]$Resume)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonExecutable = 'D:\softwares\Anaconda3\envs\astar-drl\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    $pythonExecutable = (Get-Command python -ErrorAction Stop).Source
}
Push-Location -LiteralPath $projectRoot
try {
    if (Test-Path -LiteralPath 'outputs/asvc_fixed_ce_five_seeds_v1/runner.lock') {
        throw 'Extension runner lock exists. Inspect the existing process before starting another.'
    }
    $otherPython = @(Get-Process python,pythonw -ErrorAction SilentlyContinue)
    if ($otherPython.Count -gt 0) {
        throw ('Existing Python work was preserved. Extension not started; PIDs: ' + (($otherPython | ForEach-Object Id) -join ','))
    }
    & $pythonExecutable -u scripts/run_fixed_ce_five.py
    if ($LASTEXITCODE -ne 0) { throw 'Extension preflight failed; no training started.' }
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $logRoot = Join-Path $projectRoot 'outputs/asvc_fixed_ce_five_seeds_v1/queue_logs'
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
    $stdout = Join-Path $logRoot ($stamp + '_stdout.log')
    $stderr = Join-Path $logRoot ($stamp + '_stderr.log')
    $arguments = @('-u','scripts/run_fixed_ce_five.py','--train','--seeds','2','3','4')
    if ($Resume) { $arguments += '--resume' }
    $process = Start-Process -FilePath $pythonExecutable -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    try { $process.PriorityClass = 'BelowNormal' } catch { Write-Warning 'Could not lower new process priority.' }
    $launch = [ordered]@{pid=$process.Id;launched=(Get-Date -Format o);new_seeds=@(2,3,4);reused_ce_seeds=@(0,1);maximum_new_runs=3;steps_per_run=200000;sequential=$true;stdout=$stdout;stderr=$stderr;resume=[bool]$Resume}
    $target = Join-Path $projectRoot 'results/asvc_fixed_ce_five_seeds_v1/launch.json'
    [IO.File]::WriteAllText($target,($launch | ConvertTo-Json -Depth 4),[Text.UTF8Encoding]::new($false))
    Write-Output ("CE extension started: PID {0}; seeds 2, 3, 4, each 200000 steps." -f $process.Id)
    Write-Output ("stdout: {0}" -f $stdout)
    Write-Output ("stderr: {0}" -f $stderr)
}
finally { Pop-Location }
