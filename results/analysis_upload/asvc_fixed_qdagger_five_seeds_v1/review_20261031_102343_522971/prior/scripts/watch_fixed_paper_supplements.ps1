param()
$ErrorActionPreference='Stop'
$projectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonExecutable='D:\softwares\Anaconda3\envs\astar-drl\python.exe'
$resultRoot=Join-Path $projectRoot 'results/asvc_fixed_paper_supplements_v1'
$statusPath=Join-Path $resultRoot 'postprocessing_status.json'
$lock=Join-Path $resultRoot 'postprocessing.lock'
function Save-Status($value) {
    $temporary=$statusPath+'.tmp'
    [IO.File]::WriteAllText($temporary,($value|ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}
New-Item -ItemType Directory -Path $resultRoot -Force | Out-Null
$lockStream=[IO.File]::Open($lock,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
$lockStream.Dispose()
Push-Location -LiteralPath $projectRoot
try {
    $sources=@('scripts/watch_fixed_paper_supplements.ps1','scripts/review_fixed_paper_supplements.py',
        'scripts/audit_fixed_paper_supplements.py','scripts/analyze_fixed_qdagger.py')
    $digests=[ordered]@{}
    foreach ($source in $sources) { $digests[$source]=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() }
    $registration=[ordered]@{pid=$PID;started=(Get-Date -Format o);source_sha256=$digests;read_only_training=$true;policy_evaluations=0;new_training_runs=0}
    [IO.File]::WriteAllText((Join-Path $resultRoot 'postprocessing_registration.json'),($registration|ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
    Save-Status ([ordered]@{running=$true;complete=$false;phase='waiting_for_existing_qdagger_queue';pid=$PID;updated=(Get-Date -Format o)})
    $deadline=(Get-Date).AddHours(24)
    while ($true) {
        $queue=Get-Content -LiteralPath 'results/asvc_fixed_qdagger_v1/queue_status.json' -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($queue.complete) { break }
        if (-not $queue.running) { throw ('QDagger queue stopped: '+$queue.error) }
        if ((Get-Date) -gt $deadline) { throw 'Postprocessing wait limit reached; training process preserved.' }
        Start-Sleep -Seconds 30
    }
    foreach ($source in $sources) {
        if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $digests[$source]) { throw ('Postprocessing source changed: '+$source) }
    }
    Save-Status ([ordered]@{running=$true;complete=$false;phase='auditing_completed_evidence';pid=$PID;updated=(Get-Date -Format o)})
    & $pythonExecutable -X utf8 -u scripts/audit_fixed_paper_supplements.py
    if ($LASTEXITCODE -ne 0) { throw 'Completed evidence audit failed; logs and old data preserved.' }
    & $pythonExecutable -X utf8 -u scripts/analyze_fixed_qdagger.py
    if ($LASTEXITCODE -ne 0) { throw 'Final QDagger summary failed.' }
    & $pythonExecutable -X utf8 -u scripts/review_fixed_paper_supplements.py --plot
    if ($LASTEXITCODE -ne 0) { throw 'Final combined review failed.' }
    Save-Status ([ordered]@{running=$false;complete=$true;phase='complete';pid=$PID;report='results/asvc_fixed_paper_supplements_v1/REPORT.zh-CN.md';updated=(Get-Date -Format o)})
}
catch {
    Save-Status ([ordered]@{running=$false;complete=$false;error=$_.Exception.Message;training_preserved=$true;updated=(Get-Date -Format o)})
    Write-Error $_
    exit 1
}
finally {
    Pop-Location
    if (Test-Path -LiteralPath $lock) { Remove-Item -LiteralPath $lock }
}
