#Requires -Version 5.1
<#
.SYNOPSIS
Install a pinned CPU environment, repair this checkout, and run a quick smoke test.
.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File .\concept-results\setup-fedbif.ps1
.EXAMPLE
.\concept-results\setup-fedbif.ps1 -PythonExe C:\Python311\python.exe -SkipInstall
#>
[CmdletBinding()]
param(
    [string]$RepoPath = '',
    [string]$PythonExe = '',
    [switch]$SkipInstall,
    [switch]$SkipSmoke,
    [switch]$ValidateOnly,
    [ValidateRange(1, 20)]
    [int]$Rounds = 4
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($RepoPath)) {
    $RepoPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'fedbif-tpds25'
}
$RepoPath = (Resolve-Path -LiteralPath $RepoPath).ProviderPath
$repairScript = Join-Path $PSScriptRoot 'repair_imports.py'
$smokeScript = Join-Path $PSScriptRoot 'quick_train.py'
$requirements = Join-Path $RepoPath 'requirements.txt'
$venvDirectory = Join-Path $RepoPath '.venv'
$venvPython = Join-Path $venvDirectory 'Scripts\python.exe'
$resultPath = Join-Path $PSScriptRoot 'training-results.json'

Write-Host "Working directory: $RepoPath"
foreach ($requiredPath in @($repairScript, $smokeScript, $requirements,
        (Join-Path $RepoPath 'plugin\bif.py'), (Join-Path $RepoPath 'plugin\models.py'))) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file missing: $requiredPath. Keep this script and its Python helpers together."
    }
}
if ($ValidateOnly) {
    $parseTokens = $null
    $parseErrors = $null
    $null = [System.Management.Automation.Language.Parser]::ParseFile(
        $PSCommandPath, [ref]$parseTokens, [ref]$parseErrors)
    if ($parseErrors.Count -gt 0) {
        throw ($parseErrors | Out-String)
    }
    Write-Host 'PowerShell syntax and required file locations OK. No installation or execution performed.'
    return
}

function Test-Python311 {
    param([string]$Executable)
    if ([string]::IsNullOrWhiteSpace($Executable) -or
        $Executable -match '\\Microsoft\\WindowsApps\\' -or
        -not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        return $null
    }
    try {
        $probe = @(& $Executable -c "import struct, sys; print(sys.executable); sys.exit(0 if sys.version_info[:2] == (3, 11) and struct.calcsize('P') == 8 else 1)" 2>$null)
        if ($LASTEXITCODE -eq 0 -and $probe.Count -gt 0) {
            return [string]$probe[-1]
        }
    } catch {
        # Inaccessible launch aliases are not usable interpreters.
    }
    return $null
}

function Find-Python311 {
    $candidates = [System.Collections.Generic.List[string]]::new()
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        try {
            $launched = @(& $launcher.Source -3.11 -c 'import sys; print(sys.executable)' 2>$null)
            if ($LASTEXITCODE -eq 0 -and $launched.Count -gt 0) {
                $candidates.Add([string]$launched[-1])
            }
        } catch { }
    }
    foreach ($registryKey in @(
        'HKCU:\Software\Python\PythonCore\3.11\InstallPath',
        'HKLM:\Software\Python\PythonCore\3.11\InstallPath')) {
        if (Test-Path -LiteralPath $registryKey) {
            $key = Get-Item -LiteralPath $registryKey
            $executableValue = $key.GetValue('ExecutablePath')
            $directoryValue = $key.GetValue('')
            if ($executableValue) { $candidates.Add([string]$executableValue) }
            if ($directoryValue) { $candidates.Add((Join-Path $directoryValue 'python.exe')) }
        }
    }
    if ($env:LOCALAPPDATA) {
        $candidates.Add((Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'))
    }
    if ($env:ProgramFiles) {
        $candidates.Add((Join-Path $env:ProgramFiles 'Python311\python.exe'))
    }
    foreach ($name in @('python.exe', 'python3.exe', 'python3.11.exe')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) { $candidates.Add($command.Source) }
    }
    foreach ($candidate in $candidates) {
        $found = Test-Python311 $candidate
        if ($found) { return $found }
    }
    return $null
}

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $($Arguments -join ' ')"
    }
}

if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $resolvedPython = Test-Python311 $venvPython
    if (-not $resolvedPython) {
        throw "Existing environment $venvDirectory must use 64-bit Python 3.11. Rename it and rerun; it has not been deleted."
    }
} else {
    if ($SkipInstall) {
        throw '-SkipInstall requires an existing .venv with 64-bit Python 3.11. Run again without -SkipInstall to set it up.'
    }
    if ($PythonExe) {
        $resolvedPython = Test-Python311 $PythonExe
        if (-not $resolvedPython) {
            throw "-PythonExe must point to a usable 64-bit Python 3.11 executable: $PythonExe"
        }
    } else {
        $resolvedPython = Find-Python311
    }
    if (-not $resolvedPython) {
        $wingetCommand = Get-Command winget.exe -ErrorAction SilentlyContinue
        if (-not $wingetCommand) {
            throw 'Python 3.11 x64 and WinGet are unavailable. Download the Windows installer (64-bit) from https://www.python.org/downloads/release/python-3119/ and install it for the current user. Rerun this script with -PythonExe pointing to the installed python.exe; see README.md for the exact command.'
        }
        Write-Host 'Installing Python 3.11 for the current user...'
        Invoke-Checked $wingetCommand.Source @('install', '--id', 'Python.Python.3.11', '--exact',
            '--source', 'winget', '--scope', 'user', '--silent',
            '--accept-package-agreements', '--accept-source-agreements')
        $resolvedPython = Find-Python311
        if (-not $resolvedPython) {
            throw 'Python installation finished but no usable 3.11 x64 interpreter was found. Open a new terminal and rerun with -PythonExe pointing to the installed python.exe.'
        }
    }
    Invoke-Checked $resolvedPython @('-m', 'venv', $venvDirectory)
    if (-not (Test-Python311 $venvPython)) {
        throw "Virtual environment creation did not produce a usable interpreter: $venvPython"
    }
}

Push-Location -LiteralPath $RepoPath
try {
    if (-not $SkipInstall) {
        Write-Host 'Installing pinned dependencies and CPU-only PyTorch (download time depends on the connection)...'
        Invoke-Checked $venvPython @('-m', 'pip', 'install', '--disable-pip-version-check', 'numpy==1.25.0')
        Invoke-Checked $venvPython @('-m', 'pip', 'install', '--disable-pip-version-check',
            '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch==2.2.0', 'torchvision==0.17.0')
        Invoke-Checked $venvPython @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', $requirements)
        Invoke-Checked $venvPython @('-m', 'pip', 'check')
    }
    Invoke-Checked $venvPython @('-c', "import torch, torchvision, flwr; print('Dependency imports OK')")
    Invoke-Checked $venvPython @('-m', 'py_compile', $repairScript, $smokeScript)
    Invoke-Checked $venvPython @($repairScript, '--repo', $RepoPath)
    Invoke-Checked $venvPython @('-c', "from plugin.models import get_model_with_config; print('Repository model imports OK')")
    if (-not $SkipSmoke) {
        Invoke-Checked $venvPython @($smokeScript, '--repo', $RepoPath, '--output', $resultPath,
            '--rounds', [string]$Rounds)
        Write-Host "Results: $resultPath"
    }
    Write-Host 'Setup complete. Use -SkipInstall on subsequent runs to reuse this environment.'
} finally {
    Pop-Location
}
