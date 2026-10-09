param(
    [switch]$Preview,
    [switch]$ValueRepairOnly,
    [string]$ProxyUrl,
    [string]$Message = 'Add eleven-method five-seed results and frozen 500-scene evaluation'
)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$gitSafeRoot = $projectRoot.Replace('\', '/')
$gitCommand = Get-Command git -ErrorAction SilentlyContinue
$gitExecutable = if ($gitCommand) { $gitCommand.Source } else { 'D:\softwares\Anaconda3\envs\yolov5\Library\cmd\git.exe' }
if (-not (Test-Path -LiteralPath $gitExecutable -PathType Leaf)) {
    throw 'Git executable was not found.'
}
$gitOptions = @('-c', "safe.directory=$gitSafeRoot", '-c', 'core.quotePath=false')
if ($ProxyUrl) { $gitOptions += @('-c', "http.proxy=$ProxyUrl") }
$scope = @('.')
if ($ValueRepairOnly) {
    $scope = @('README.md', '.gitattributes', '.gitignore', 'docs/VALUE_REPAIR_V1.zh-CN.md', 'docs/GITHUB_SYNC.zh-CN.md',
               'scripts/analyze_value_repair.py', 'scripts/inspect_value_repair_navigation.py',
               'scripts/package_value_repair_analysis.py', 'scripts/sync_github.ps1',
               'scripts/run_value_repair.py', 'configs/whole_map_91701_value_repair_v1.yaml',
               'configs/whole_map_91701_value_repair_pilot_v1.yaml', 'src/astar_d3qn',
               'results/whole_map_91701_value_repair_v1', 'results/analysis_upload',
               'configs/whole_map_91701_teaching_efficiency_v1.yaml',
               'docs/TEACHING_EFFICIENCY_V1.zh-CN.md', 'scripts/teaching_efficiency_common.py',
               'scripts/prepare_teaching_efficiency.py', 'scripts/run_teaching_efficiency.py',
               'scripts/analyze_teaching_efficiency.py', 'scripts/evaluate_teaching_efficiency.py',
               'scripts/review_teaching_efficiency_results.py',
               'scripts/package_teaching_efficiency_analysis.py',
               'tests/test_teaching_efficiency.py', 'data/whole_map_91701_teaching_efficiency_v1',
               'results/whole_map_91701_teaching_efficiency_v1',
               'configs/whole_map_91701_supervision_tail_v1.yaml',
               'docs/SUPERVISION_TAIL_V1.zh-CN.md', 'scripts/supervision_tail_common.py',
               'scripts/prepare_supervision_tail.py', 'scripts/run_supervision_tail.py',
               'scripts/analyze_supervision_tail.py', 'scripts/evaluate_supervision_tail.py',
               'tests/test_supervision_tail.py', 'data/whole_map_91701_supervision_tail_v1',
               'results/whole_map_91701_supervision_tail_v1',
               'scripts/review_supervision_tail_partial.py',
               'configs/whole_map_91701_dqfd_comparison_v1.yaml',
               'docs/DQFD_COMPARISON_V1.zh-CN.md', 'scripts/dqfd_comparison_common.py',
               'scripts/prepare_dqfd_comparison.py', 'scripts/run_dqfd_comparison.py',
               'scripts/summarize_dqfd_comparison.py', 'scripts/package_dqfd_comparison_analysis.py',
               'tests/test_dqfd_comparison.py', 'data/whole_map_91701_dqfd_comparison_v1',
               'results/whole_map_91701_dqfd_comparison_v1')
}

function Invoke-ProjectGit {
    & $gitExecutable @gitOptions --literal-pathspecs @args
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
        Invoke-ProjectGit ls-files -ci --exclude-standard -- @scope
        Write-Output 'Changes within the selected upload scope:'
        Invoke-ProjectGit status --short -- @scope
        Write-Output 'No index changes, commit, push or training performed.'
        return
    }
    if ($ValueRepairOnly) {
        # A commit includes every staged file. Preserve other staged work by stopping first.
        $outsideScope = @(Invoke-ProjectGit diff --cached --name-only | Where-Object {
            $stagedPath = $_
            -not ($scope | Where-Object { $stagedPath -eq $_ -or $stagedPath.StartsWith($_ + '/') })
        })
        if ($outsideScope.Count -gt 0) {
            throw ('Unrelated staged files would enter the commit; preserve them and review first: ' + ($outsideScope -join ', '))
        }
    }

    # NUL-separated paths preserve spaces and Unicode; only remove matching index entries.
    $pathList = Join-Path ([IO.Path]::GetTempPath()) ('astar-github-ignored-' + [guid]::NewGuid().ToString('N') + '.txt')
    try {
        $ignoredPaths = @(Invoke-ProjectGit ls-files -ci --exclude-standard -- @scope)
        if ($ignoredPaths.Count -gt 0) {
            $utf8 = New-Object System.Text.UTF8Encoding($false)
            [IO.File]::WriteAllText($pathList, (($ignoredPaths -join "`0") + "`0"), $utf8)
            Invoke-ProjectGit rm --cached --ignore-unmatch "--pathspec-from-file=$pathList" --pathspec-file-nul
        }
    }
    finally {
        if (Test-Path -LiteralPath $pathList) { Remove-Item -LiteralPath $pathList }
    }
    Invoke-ProjectGit add --all -- @scope
    & $gitExecutable @gitOptions diff --cached --quiet
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
