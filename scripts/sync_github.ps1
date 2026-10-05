param(
    [switch]$Preview,
    [string]$Message = 'Update dynamic navigation experiments and five-seed path guidance analysis'
)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$gitSafeRoot = $projectRoot.Replace('\', '/')
$gitCommand = Get-Command git -ErrorAction SilentlyContinue
$gitExecutable = if ($gitCommand) { $gitCommand.Source } else { 'D:\softwares\Anaconda3\envs\yolov5\Library\cmd\git.exe' }
if (-not (Test-Path -LiteralPath $gitExecutable -PathType Leaf)) {
    throw 'Git executable was not found.'
}

function Invoke-ProjectGit {
    & $gitExecutable -c "safe.directory=$gitSafeRoot" -c core.quotePath=false --literal-pathspecs @args
    if ($LASTEXITCODE -ne 0) {
        throw "Git command failed: $($args[0]) (exit $LASTEXITCODE)."
    }
}

Push-Location -LiteralPath $projectRoot
try {
    $branch = (Invoke-ProjectGit branch --show-current).Trim()
    $remote = (Invoke-ProjectGit remote get-url --push origin).Trim()
    if ($branch -ne 'main') { throw "Expected main branch; found $branch." }
    if ($remote -notin @('https://github.com/Dylan923057/astar-d3qn-dynamic-navigation.git',
                        'git@github.com:Dylan923057/astar-d3qn-dynamic-navigation.git')) {
        throw 'The origin push URL does not match the reviewed repository.'
    }
    if ($Preview) {
        Write-Output 'Tracked files to stop tracking (local files will remain):'
        Invoke-ProjectGit ls-files -ci --exclude-standard
        Write-Output 'Current project changes:'
        Invoke-ProjectGit status --short
        Write-Output 'No index changes, commit, push or training performed.'
        return
    }

    # NUL-separated paths preserve spaces and Unicode; only remove matching index entries.
    $pathList = Join-Path ([IO.Path]::GetTempPath()) ('astar-github-ignored-' + [guid]::NewGuid().ToString('N') + '.txt')
    try {
        $ignoredPaths = @(Invoke-ProjectGit ls-files -ci --exclude-standard)
        if ($ignoredPaths.Count -gt 0) {
            $utf8 = New-Object System.Text.UTF8Encoding($false)
            [IO.File]::WriteAllText($pathList, (($ignoredPaths -join "`0") + "`0"), $utf8)
            Invoke-ProjectGit rm --cached --ignore-unmatch "--pathspec-from-file=$pathList" --pathspec-file-nul
        }
    }
    finally {
        if (Test-Path -LiteralPath $pathList) { Remove-Item -LiteralPath $pathList }
    }
    Invoke-ProjectGit add --all
    & $gitExecutable -c "safe.directory=$gitSafeRoot" diff --cached --quiet
    $diffCode = $LASTEXITCODE
    if ($diffCode -eq 1) {
        Invoke-ProjectGit commit -m $Message
    }
    elseif ($diffCode -ne 0) {
        throw "Checking staged changes failed (exit $diffCode)."
    }
    Invoke-ProjectGit push origin main
}
finally {
    Pop-Location
}
