param(
    [switch]$Preview,
    [string]$ProxyUrl,
    [string]$Message = 'Archive five-seed fixed-task CE comparison and complete validation evidence'
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$gitCommand = Get-Command git -ErrorAction SilentlyContinue
$gitExecutable = if ($gitCommand) { $gitCommand.Source } else { 'D:\softwares\Anaconda3\envs\yolov5\Library\cmd\git.exe' }
if (-not (Test-Path -LiteralPath $gitExecutable -PathType Leaf)) { throw 'Git executable was not found.' }
$pythonExecutable = 'D:\softwares\Anaconda3\envs\astar-drl\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) { $pythonExecutable = (Get-Command python -ErrorAction Stop).Source }
$gitOptions = @('-c', ('safe.directory=' + $projectRoot.Replace('\','/')), '-c', 'core.quotePath=false')
if ($ProxyUrl) { $gitOptions += @('-c', "http.proxy=$ProxyUrl") }
else { $gitOptions += @('-c', 'http.proxy=') }
function Invoke-ProjectGit {
    & $gitExecutable @gitOptions --literal-pathspecs @args
    if ($LASTEXITCODE -ne 0) { throw "Git command failed: $($args[0]) (exit $LASTEXITCODE)." }
}
Push-Location -LiteralPath $projectRoot
try {
    $branch = (Invoke-ProjectGit branch --show-current).Trim()
    $remote = (Invoke-ProjectGit remote get-url --push origin).Trim()
    if ($branch -ne 'main') { throw "Expected main; found $branch." }
    if ($remote -notin @('https://github.com/Dylan923057/astar-d3qn-dynamic-navigation.git','git@github.com:Dylan923057/astar-d3qn-dynamic-navigation.git')) { throw 'Origin does not match the reviewed repository.' }
    $bundleRoot = Join-Path $projectRoot 'results/analysis_upload/asvc_fixed_ce_five_seeds_v1'
    $pointer = Get-Content -LiteralPath (Join-Path $bundleRoot 'latest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $bundlePath = [IO.Path]::GetFullPath((Join-Path $projectRoot $pointer.package))
    if (-not $bundlePath.StartsWith($bundleRoot + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Bundle path escapes this batch.' }
    & $pythonExecutable -X utf8 scripts/package_fixed_ce_five_evidence.py --verify --check-sources
    if ($LASTEXITCODE -ne 0) { throw 'Package/source verification failed; nothing staged or pushed.' }
    $manifestPath = Join-Path $bundlePath 'manifest.json'
    if ((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $pointer.manifest_sha256) { throw 'Publication pointer hash differs.' }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $scope = Get-Content -LiteralPath (Join-Path $bundlePath 'upload_scope.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $scope = @($scope | ForEach-Object { [string]$_ })
    foreach ($item in $scope) {
        $resolved = [IO.Path]::GetFullPath((Join-Path $projectRoot $item))
        if (-not $resolved.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -or
            $item -eq '.' -or $item.StartsWith('.git/') -or $item.StartsWith('outputs/') -or
            -not (Test-Path -LiteralPath $resolved)) { throw "Invalid upload path: $item" }
    }
    $files = @(Invoke-ProjectGit ls-files --cached --others --exclude-standard -- @scope | Sort-Object -Unique)
    $bundleRelative = $pointer.package.Replace('\','/')
    $required = @($manifest.files_sha256.PSObject.Properties | ForEach-Object { "$bundleRelative/$($_.Name)" }) + "$bundleRelative/manifest.json"
    foreach ($item in $required) {
        if ($item -notin $files) { throw "A required bundle file is ignored/missing: $item" }
    }
    [long]$totalBytes = 0
    foreach ($item in $files) {
        $info = Get-Item -LiteralPath (Join-Path $projectRoot $item)
        if ($info.PSIsContainer -or $info.Length -ge 50MB -or $info.Extension -in @('.pth','.pt','.zip','.exe','.dll','.npz') -or
            $item -match '(__pycache__|matplotlib_cache|checkpoint_latest)') { throw "Unexpected large/model/cache artifact: $item" }
        $totalBytes += $info.Length
    }
    Write-Output ("Reviewed scope: {0} files, {1:N2} MiB. 5 CE vs 5 margin seeds; 3 new CE runs; 10500 validation records; final500 unused." -f $files.Count,($totalBytes/1MB))
    $outside = @(Invoke-ProjectGit diff --cached --name-only | Where-Object {
        $stagedPath = $_
        -not ($scope | Where-Object { $stagedPath -eq $_ -or $stagedPath.StartsWith($_ + '/') })
    })
    if ($outside.Count -gt 0) { throw ('Unrelated staged files preserved; review first: ' + ($outside -join ', ')) }
    if ($Preview) {
        Invoke-ProjectGit status --short -- @scope
        Write-Output 'Preview complete: no network, staging, commit, push, training or policy evaluation.'
        return
    }
    # Read remote before modifying the index. Do not push unrelated local commits.
    $remoteLine = @(Invoke-ProjectGit ls-remote --exit-code --heads origin refs/heads/main)
    if ($remoteLine.Count -ne 1) { throw 'Cannot identify a unique remote main.' }
    $remoteHead = ($remoteLine[0] -split '\s+')[0]
    $localHead = (Invoke-ProjectGit rev-parse HEAD).Trim()
    if ($localHead -ne $remoteHead) {
        $parent = (Invoke-ProjectGit rev-parse 'HEAD^').Trim()
        if ($parent -ne $remoteHead) { throw 'Remote main differs. Review history manually; no automatic merge or force push.' }
        $pendingFiles = @(Invoke-ProjectGit diff-tree --no-commit-id --name-only -r HEAD)
        $unrelatedPending = @($pendingFiles | Where-Object {
            $pendingPath = $_
            -not ($scope | Where-Object { $pendingPath -eq $_ -or $pendingPath.StartsWith($_ + '/') })
        })
        if ($unrelatedPending.Count -gt 0) { throw 'Unrelated unpublished commit detected; no automatic push.' }
    }
    Invoke-ProjectGit add --all -- @scope
    & $gitExecutable @gitOptions diff --cached --quiet
    $diffCode = $LASTEXITCODE
    if ($diffCode -eq 1) { Invoke-ProjectGit commit -m $Message }
    elseif ($diffCode -ne 0) { throw "Checking staged changes failed (exit $diffCode)." }
    Invoke-ProjectGit push origin main
    Write-Output 'Upload complete. Five paired seeds archived; only CE seeds 2/3/4 were newly trained.'
}
finally { Pop-Location }
