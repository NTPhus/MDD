<#
.SYNOPSIS
Run native Windows CUDA training with explicit, resumable run IDs.
.EXAMPLE
pwsh -File scripts/run_local_training.ps1 smoke -RunId windows-smoke-01
.EXAMPLE
pwsh -File scripts/run_local_training.ps1 A -RunId windows-A-seed42
.EXAMPLE
pwsh -File scripts/run_local_training.ps1 B -RunId windows-B-fold1 -Fold 1 -InitCheckpoint runs/windows-A-seed42/best_checkpoint
.EXAMPLE
pwsh -File scripts/run_local_training.ps1 A -RunId windows-A-seed42 -Resume runs/windows-A-seed42/checkpoint
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)][ValidateSet('smoke', 'A', 'B', 'C')][string]$Stage,
    [string]$RunId,
    [ValidateRange(1, 4)][int]$Fold,
    [string]$InitCheckpoint,
    [string]$Resume,
    [string]$RunsRoot = 'runs',
    [int]$Seed,
    [ValidateRange(1, 2147483647)][int]$MaxSteps,
    [string[]]$TrainingArguments = @(),
    [switch]$DryRun,
    [switch]$Help
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$TaskRoot = Split-Path -Parent $PSScriptRoot
if ($Help) {
    Write-Host @'
Usage: pwsh -File scripts/run_local_training.ps1 smoke|A|B|C -RunId ID [options]

smoke: One full-backbone optimizer step on a technical subset; no benchmark.
A: Source training, 2000 optimizer steps (configs/windows_source.json).
B: Vietnamese adaptation, 400 optimizer steps (configs/windows_adapt_vi.json).
C: Matched source-only control, 400 steps (configs/windows_control.json).

Options: -Fold 1..4, -InitCheckpoint PATH, -Resume PATH, -RunsRoot PATH,
-Seed N, -MaxSteps N, -DryRun. Additional Python flags may be supplied as
-TrainingArguments @('--max-train-samples', '2', '--purpose', 'technical_smoke_only').
The launcher selects CUDA and the Windows config. Use train_mdd.py directly
to choose a different device/config. Run IDs are required; use -Resume for
continuation. Stages B/C require the completed scientific stage-A checkpoint.
'@
    return
}
if (-not $Stage) { throw 'Choose stage smoke, A, B or C. Use -Help for examples.' }
if ([string]::IsNullOrWhiteSpace($RunId)) { throw 'Choose an explicit -RunId; use -Resume to continue an existing run.' }
if ($InitCheckpoint -and $Resume) { throw 'Use either -InitCheckpoint or -Resume.' }

$configNames = @{ smoke = 'windows_smoke.json'; A = 'windows_source.json'; B = 'windows_adapt_vi.json'; C = 'windows_control.json' }
$configPath = Join-Path $TaskRoot "configs/$($configNames[$Stage])"
$VenvPython = Join-Path $TaskRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) { throw "Missing Windows config: $configPath" }
foreach ($argument in $TrainingArguments) {
    if ($argument -match '^--(config|device|run-id)(=|$)') {
        throw 'Use launcher parameters for run IDs; the CUDA device and Windows config are fixed.'
    }
}
$pythonArguments = @('-u', (Join-Path $TaskRoot 'scripts/train_mdd.py'), '--run-id', $RunId, '--runs-root', $RunsRoot)
if ($PSBoundParameters.ContainsKey('Fold')) { $pythonArguments += @('--fold', "$Fold") }
if ($InitCheckpoint) { $pythonArguments += @('--init-checkpoint', $InitCheckpoint) }
if ($Resume) { $pythonArguments += @('--resume', $Resume) }
if ($PSBoundParameters.ContainsKey('Seed')) { $pythonArguments += @('--seed', "$Seed") }
if ($PSBoundParameters.ContainsKey('MaxSteps')) { $pythonArguments += @('--max-steps', "$MaxSteps") }
$pythonArguments += $TrainingArguments
$pythonArguments += @('--config', $configPath, '--device', 'cuda')
if ($DryRun) {
    $quotedCommand = @($VenvPython) + $pythonArguments | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }
    Write-Host ('& ' + ($quotedCommand -join ' '))
    return
}
if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw 'Missing Windows .venv. Run: pwsh -File scripts/setup_local_training.ps1'
}
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
Push-Location -LiteralPath $TaskRoot
try {
    Write-Host "Starting stage $Stage with configs/$($configNames[$Stage]) on CUDA."
    & $VenvPython @pythonArguments
    if ($LASTEXITCODE -ne 0) { throw "Training stopped (exit $LASTEXITCODE). Preserve the run and inspect its logs before resuming." }
}
finally {
    Pop-Location
}
