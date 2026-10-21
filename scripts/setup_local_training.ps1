<#
.SYNOPSIS
Prepare the native Windows CUDA environment without starting training.
.EXAMPLE
pwsh -File scripts/setup_local_training.ps1
.EXAMPLE
pwsh -File scripts/setup_local_training.ps1 -Offline -SkipDataPreparation
#>
[CmdletBinding()]
param(
    [string]$PythonExecutable,
    [switch]$Offline,
    [switch]$SkipDataPreparation,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$TaskRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $TaskRoot '.venv/Scripts/python.exe'
$TorchVersion = '2.7.1+cu126'
$TorchIndex = 'https://download.pytorch.org/whl/cu126'

function Invoke-TrainingPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $VenvPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed (exit $LASTEXITCODE): $($Arguments -join ' ')"
    }
}

function Find-BootstrapPython {
    if ($PythonExecutable) {
        if (Test-Path -LiteralPath $PythonExecutable -PathType Leaf) {
            return (Resolve-Path -LiteralPath $PythonExecutable).Path
        }
        $selectedCommand = Get-Command $PythonExecutable -CommandType Application -ErrorAction SilentlyContinue
        if ($selectedCommand) { return $selectedCommand.Source }
        throw "Python executable does not exist: $PythonExecutable"
    }
    $managedRoot = Join-Path $TaskRoot '.tools/python'
    if (Test-Path -LiteralPath $managedRoot -PathType Container) {
        $managedPython = Get-ChildItem -LiteralPath $managedRoot -Filter python.exe -File -Recurse |
            Where-Object { $_.FullName -notmatch '[\\/]Scripts[\\/]' } |
            Sort-Object FullName |
            Select-Object -First 1
        if ($managedPython) { return $managedPython.FullName }
    }
    foreach ($candidate in @('python.exe', 'python3.exe')) {
        $selectedCommand = Get-Command $candidate -CommandType Application -ErrorAction SilentlyContinue
        if ($selectedCommand -and $selectedCommand.Source -notmatch '[\\/]WindowsApps[\\/]') {
            return $selectedCommand.Source
        }
    }
    if ($env:LOCALAPPDATA) {
        foreach ($version in @('312', '311', '310', '313')) {
            $candidate = Join-Path $env:LOCALAPPDATA "Programs/Python/Python$version/python.exe"
            if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
        }
    }
    throw 'Python 3.10-3.13 is required. Provide -PythonExecutable with its full path.'
}

$RequirementsProbe = @'
import importlib.metadata
from pathlib import Path
import sys

for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    requirement = line.split("#", 1)[0].strip()
    if not requirement:
        continue
    name, separator, expected = requirement.partition("==")
    if not separator:
        raise SystemExit(f"Expected a pinned requirement: {requirement}")
    try:
        actual = importlib.metadata.version(name.strip())
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit(1)
    if actual != expected.strip():
        raise SystemExit(1)
'@

Push-Location -LiteralPath $TaskRoot
try {
    if ($DryRun) {
        Write-Host "Environment: $VenvPython (Python 3.10-3.13, 64-bit)"
        Write-Host "PyTorch: torch==$TorchVersion from $TorchIndex"
        Write-Host 'Dependencies: requirements/training.txt; matching pins are reused.'
        if ($Offline) { Write-Host 'Offline: require installed dependencies and verified local model files.' }
        else { Write-Host 'Download/verify: scripts/download_models.py --models wav2vec2-base' }
        if (-not $SkipDataPreparation) { Write-Host 'Data: scripts/prepare_l2arctic.py --verify-audio' }
        Write-Host 'Check: scripts/check_training_environment.py --require-local-cuda --output data/generated/setup/windows-cuda.json'
        return
    }

    if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
        $bootstrapPython = Find-BootstrapPython
        & $bootstrapPython -c 'import struct, sys; assert (3, 10) <= sys.version_info[:2] <= (3, 13), "Use Python 3.10-3.13"; assert struct.calcsize("P") == 8, "Use 64-bit Python"'
        if ($LASTEXITCODE -ne 0) { throw 'The selected Python is not supported by these dependency pins.' }
        & $bootstrapPython -m venv (Join-Path $TaskRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the local .venv.' }
    }

    $TorchProbe = @'
import importlib.metadata
import sys
try:
    installed = importlib.metadata.version("torch")
except importlib.metadata.PackageNotFoundError:
    installed = None
raise SystemExit(0 if installed == sys.argv[1] else 1)
'@
    & $VenvPython -c $TorchProbe $TorchVersion
    $torchMatches = $LASTEXITCODE -eq 0
    if (-not $torchMatches) {
        if ($Offline) { throw "Offline setup requires torch==$TorchVersion already installed in .venv." }
        Invoke-TrainingPython -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '--retries', '2', '--timeout', '60', '--index-url', $TorchIndex, "torch==$TorchVersion")
    }
    else { Write-Host "Reusing torch==$TorchVersion." }

    & $VenvPython -c $RequirementsProbe 'requirements/training.txt'
    $trainingPinsMatch = $LASTEXITCODE -eq 0
    if (-not $trainingPinsMatch) {
        if ($Offline) { throw 'Offline setup requires all pins from requirements/training.txt already installed.' }
        Invoke-TrainingPython -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '--retries', '2', '--timeout', '60', '--index-url', 'https://pypi.org/simple', '-r', 'requirements/training.txt')
    }
    else { Write-Host 'Reusing the pinned training dependencies.' }
    Invoke-TrainingPython -Arguments @('-m', 'pip', 'check')
    $freeze = & $VenvPython -m pip freeze
    if ($LASTEXITCODE -ne 0) { throw 'Could not freeze the installed dependencies.' }
    [System.IO.File]::WriteAllLines((Join-Path $TaskRoot 'requirements/local-windows-cuda.lock.txt'), [string[]]$freeze, (New-Object System.Text.UTF8Encoding($false)))

    $env:OMP_NUM_THREADS = '4'
    $env:MKL_NUM_THREADS = '4'
    $env:TOKENIZERS_PARALLELISM = 'false'
    Invoke-TrainingPython -Arguments @('-c', 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable; check the NVIDIA driver and CUDA PyTorch wheel"; x = torch.ones(16, device="cuda", requires_grad=True); x.square().sum().backward(); torch.cuda.synchronize(); print("CUDA backward OK:", torch.cuda.get_device_name(0), "runtime", torch.version.cuda)')

    $modelManifest = Join-Path $TaskRoot 'models/pretrained/wav2vec2-base/download_manifest.json'
    $modelArguments = @('scripts/download_models.py', '--models', 'wav2vec2-base')
    if ($Offline) { $modelArguments += '--verify-only' }
    Invoke-TrainingPython -Arguments $modelArguments

    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    if (-not $SkipDataPreparation) {
        Invoke-TrainingPython -Arguments @('scripts/prepare_l2arctic.py', '--verify-audio')
    }
    & $VenvPython 'scripts/check_training_environment.py' '--require-local-cuda' '--output' 'data/generated/setup/windows-cuda.json'
    if ($LASTEXITCODE -ne 0) {
        if ($SkipDataPreparation) {
            Write-Warning 'CUDA dependencies and model were verified, but training inputs are incomplete. Inspect data/generated/setup/windows-cuda.json and prepare the raw corpus before training.'
            return
        }
        throw 'Training readiness check failed. Inspect data/generated/setup/windows-cuda.json.'
    }
    Write-Host 'Windows CUDA setup is ready. No training was started.'
    Write-Host 'Next: pwsh -File scripts/run_local_training.ps1 smoke -RunId windows-smoke-01'
}
finally {
    Pop-Location
}
