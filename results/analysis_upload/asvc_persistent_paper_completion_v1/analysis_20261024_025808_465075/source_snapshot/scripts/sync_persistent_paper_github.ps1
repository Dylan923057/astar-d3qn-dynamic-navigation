param(
    [switch]$Preview,
    [string]$ProxyUrl,
    [string]$Message = 'Add persistent ASVC module ablations and density evaluation'
)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$gitCommand = Get-Command git -ErrorAction SilentlyContinue
$gitExecutable = if ($gitCommand) { $gitCommand.Source } else { 'D:\softwares\Anaconda3\envs\yolov5\Library\cmd\git.exe' }
if (-not (Test-Path -LiteralPath $gitExecutable -PathType Leaf)) { throw 'Git executable was not found.' }
$pythonExecutable = 'D:\softwares\Anaconda3\envs\astar-drl\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    $pythonCommand = Get-Command python -ErrorAction Stop
    $pythonExecutable = $pythonCommand.Source
}
$gitOptions = @('-c', ('safe.directory=' + $projectRoot.Replace('\', '/')), '-c', 'core.quotePath=false')
if ($ProxyUrl) { $gitOptions += @('-c', "http.proxy=$ProxyUrl") }

function Invoke-ProjectGit {
    & $gitExecutable @gitOptions --literal-pathspecs @args
    if ($LASTEXITCODE -ne 0) { throw "Git command failed: $($args[0]) (exit $LASTEXITCODE)." }
}

Push-Location -LiteralPath $projectRoot
try {
    $branch = (Invoke-ProjectGit branch --show-current).Trim()
    $remote = (Invoke-ProjectGit remote get-url --push origin).Trim()
    if ($branch -ne 'main') { throw "Expected main; found $branch." }
    if ($remote -notin @('https://github.com/Dylan923057/astar-d3qn-dynamic-navigation.git',
                        'git@github.com:Dylan923057/astar-d3qn-dynamic-navigation.git')) {
        throw 'Origin does not match the reviewed repository.'
    }
    $bundleRoot = Join-Path $projectRoot 'results/analysis_upload/asvc_persistent_paper_completion_v1'
    $pointer = Get-Content -LiteralPath (Join-Path $bundleRoot 'latest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $bundlePath = [IO.Path]::GetFullPath((Join-Path $projectRoot $pointer.package))
    if (-not $bundlePath.StartsWith($bundleRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Bundle path must stay within its dedicated directory.'
    }
    & $pythonExecutable -X utf8 scripts/package_persistent_paper_completion.py --verify --check-sources
    if ($LASTEXITCODE -ne 0) { throw 'Bundle verification failed; no staging, commit or push performed.' }
    $scope = (Get-Content -LiteralPath (Join-Path $bundlePath 'upload_scope.json') -Raw -Encoding UTF8 | ConvertFrom-Json)
    foreach ($scopePath in $scope) {
        $resolvedScope = [IO.Path]::GetFullPath((Join-Path $projectRoot $scopePath))
        if (-not $resolvedScope.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
            $scopePath -eq '.' -or $scopePath.StartsWith('.git/') -or -not (Test-Path -LiteralPath $resolvedScope)) {
            throw "Invalid or missing upload path: $scopePath"
        }
    }
    if ($Preview) {
        Write-Output 'Changes within this upload scope:'
        Invoke-ProjectGit status --short -- @scope
        Write-Output 'Preview complete. No staging, commit, push, training or evaluation performed.'
        return
    }
    $outside = @(Invoke-ProjectGit diff --cached --name-only | Where-Object {
        $stagedPath = $_
        -not ($scope | Where-Object { $stagedPath -eq $_ -or $stagedPath.StartsWith($_ + '/') })
    })
    if ($outside.Count -gt 0) {
        throw ('Unrelated staged files would enter this commit. Their staging is preserved: ' + ($outside -join ', '))
    }
    Invoke-ProjectGit add --all -- @scope
    & $gitExecutable @gitOptions diff --cached --quiet
    $diffCode = $LASTEXITCODE
    if ($diffCode -eq 1) { Invoke-ProjectGit commit -m $Message }
    elseif ($diffCode -ne 0) { throw "Checking staged changes failed (exit $diffCode)." }
    Invoke-ProjectGit push origin main
}
finally {
    Pop-Location
}
