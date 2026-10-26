param(
    [switch]$Preview,
    [string]$ProxyUrl,
    [string]$Message = 'Add four-run multitask ASVC risk supervision pilot evidence'
)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$gitCommand = Get-Command git -ErrorAction SilentlyContinue
$gitExecutable = if ($gitCommand) { $gitCommand.Source } else { 'D:\softwares\Anaconda3\envs\yolov5\Library\cmd\git.exe' }
if (-not (Test-Path -LiteralPath $gitExecutable -PathType Leaf)) { throw 'Git executable was not found.' }
$pythonExecutable = 'D:\softwares\Anaconda3\envs\astar-drl\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    $pythonExecutable = (Get-Command python -ErrorAction Stop).Source
}
$gitOptions = @('-c', ('safe.directory=' + $projectRoot.Replace('\','/')), '-c', 'core.quotePath=false')
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
    if ($remote -notin @('https://github.com/Dylan923057/astar-d3qn-dynamic-navigation.git', 'git@github.com:Dylan923057/astar-d3qn-dynamic-navigation.git')) {
        throw 'Origin does not match the reviewed repository.'
    }
    $bundleRoot = Join-Path $projectRoot 'results/analysis_upload/asvc_multitask_risk_pilot_v1'
    $pointer = Get-Content -LiteralPath (Join-Path $bundleRoot 'latest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $bundlePath = [IO.Path]::GetFullPath((Join-Path $projectRoot $pointer.package))
    if (-not $bundlePath.StartsWith($bundleRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Bundle path escapes this batch.'
    }
    & $pythonExecutable -X utf8 scripts/package_multitask_evidence.py --verify --check-sources
    if ($LASTEXITCODE -ne 0) { throw 'Package/source verification failed; no staging, commit or push performed.' }
    $scope = (Get-Content -LiteralPath (Join-Path $bundlePath 'upload_scope.json') -Raw -Encoding UTF8 | ConvertFrom-Json)
    $scope = @($scope | ForEach-Object { [string]$_ })
    foreach ($item in $scope) {
        $resolved = [IO.Path]::GetFullPath((Join-Path $projectRoot $item))
        if (-not $resolved.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
            $item -eq '.' -or $item.StartsWith('.git/') -or $item.StartsWith('outputs/') -or
            -not (Test-Path -LiteralPath $resolved)) { throw "Invalid upload path: $item" }
    }
    $files = @(Invoke-ProjectGit ls-files --cached --others --exclude-standard -- @scope | Sort-Object -Unique)
    [long]$totalBytes = 0
    foreach ($item in $files) {
        $path = Join-Path $projectRoot $item
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing upload file: $item" }
        $info = Get-Item -LiteralPath $path
        if ($info.Length -ge 50MB -or $info.Extension -in @('.pth','.pt','.npz','.zip') -or $item -match '(__pycache__|matplotlib_cache|preflight_recovery)') {
            throw "Unexpected large/binary/cache artifact in upload scope: $item"
        }
        $totalBytes += $info.Length
    }
    Write-Output ("Reviewed upload scope: {0} files, {1:N2} MiB." -f $files.Count, ($totalBytes / 1MB))
    if ($Preview) {
        Invoke-ProjectGit status --short -- @scope
        Write-Output 'Preview complete: no staging, commit, push, training or evaluation performed.'
        return
    }
    $outside = @(Invoke-ProjectGit diff --cached --name-only | Where-Object {
        $stagedPath = $_
        -not ($scope | Where-Object { $stagedPath -eq $_ -or $stagedPath.StartsWith($_ + '/') })
    })
    if ($outside.Count -gt 0) { throw ('Unrelated staged files are preserved; review them before this upload: ' + ($outside -join ', ')) }
    Invoke-ProjectGit add --all -- @scope
    & $gitExecutable @gitOptions diff --cached --quiet
    $diffCode = $LASTEXITCODE
    if ($diffCode -eq 1) { Invoke-ProjectGit commit -m $Message }
    elseif ($diffCode -ne 0) { throw "Checking staged changes failed (exit $diffCode)." }
    Invoke-ProjectGit push origin main
}
finally { Pop-Location }
