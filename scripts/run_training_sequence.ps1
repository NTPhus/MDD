<#
.SYNOPSIS
Run scientific Windows CUDA stages A, B, and C sequentially without changing budgets.
.DESCRIPTION
Uses the existing Windows configs and run_local_training.ps1. Stage A trains for
2000 optimizer steps; B and C each train for 400 steps from the same completed
best stage-A checkpoint. No sample/audio limits or technical-smoke overrides.
State is written atomically to StateDir/state.json. Redirect the launcher output
to StateDir/stdout.log and StateDir/stderr.log when starting it in the background.
An exclusive experiment lock prevents simultaneous sequences across folds.
Existing completed runs are verified and reused. Incomplete runs require
-ResumeExisting and a valid same-run optimizer/RNG checkpoint; no existing run
is deleted or silently restarted.
.EXAMPLE
pwsh -File scripts/run_training_sequence.ps1 -DryRun
.EXAMPLE
pwsh -File scripts/run_training_sequence.ps1 -Experiment windows-notebook-seed42 -Seed 42 -Fold 1
.EXAMPLE
pwsh -File scripts/run_training_sequence.ps1 -Experiment windows-notebook-seed42 -ResumeExisting
#>
[CmdletBinding()]
param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$')]
    [string]$Experiment = 'windows-notebook-seed42',
    [ValidateRange(0, 2147483647)][int]$Seed = 42,
    [ValidateRange(1, 4)][int]$Fold = 1,
    [string]$StateDir,
    [switch]$DryRun,
    [switch]$ResumeExisting
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$TaskRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$VenvPython = Join-Path $TaskRoot '.venv/Scripts/python.exe'
$Launcher = Join-Path $PSScriptRoot 'run_local_training.ps1'
$TrainingRoot = Join-Path $TaskRoot 'data/generated/training'
if ($Experiment -match '^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$') {
    throw 'Experiment must be a single safe directory name, not a Windows reserved device name.'
}
if ([string]::IsNullOrWhiteSpace($StateDir)) { $StateDir = Join-Path $TrainingRoot "$Experiment-fold$Fold" }
elseif (-not [IO.Path]::IsPathRooted($StateDir)) { $StateDir = Join-Path $TaskRoot $StateDir }
$StateDir = [IO.Path]::GetFullPath($StateDir)
$workspacePrefix = $TaskRoot.TrimEnd([char[]]'\/') + [IO.Path]::DirectorySeparatorChar
if (-not $StateDir.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'StateDir must be a directory inside this workspace.'
}
$stateRelative = [IO.Path]::GetRelativePath($TaskRoot, $StateDir)
if ($stateRelative -match '^\.(git|agents|codex|aws)([\/]|$)') {
    throw 'StateDir must be outside workspace metadata directories.'
}
$RunIds = [ordered]@{ A = "$Experiment-A"; B = "$Experiment-B-fold$Fold"; C = "$Experiment-C-fold$Fold" }
$ConfigNames = @{ A = 'windows_source.json'; B = 'windows_adapt_vi.json'; C = 'windows_control.json' }
$BestA = Join-Path $TaskRoot "runs/$($RunIds.A)/best_checkpoint"
$StatePath = Join-Path $StateDir 'state.json'

$RunProbe = @'
import ast
import hashlib
import json
from pathlib import Path
import sys

root, stage, run_id, seed, fold, allow_resume = sys.argv[1:]
root = Path(root)
source = ast.parse((root / "src/mdd/runtime.py").read_text(encoding="utf-8"))
defaults = next(ast.literal_eval(node.value) for node in source.body
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DEFAULTS" for t in node.targets))
name = {"A": "windows_source.json", "B": "windows_adapt_vi.json", "C": "windows_control.json"}[stage]
expected = dict(defaults, **json.loads((root / "configs" / name).read_text(encoding="utf-8")))
expected["seed"] = int(seed)
if stage in {"B", "C"}:
    expected["fold"] = int(fold)
assert expected["stage"] == stage and expected["purpose"] == "scientific_training", "Scientific stage config required"
assert expected["device"] == "cuda" and expected["mixed_precision"] == "fp16", "Windows CUDA/fp16 config required"
assert expected["max_steps"] == (2000 if stage == "A" else 400), "Scientific step budget changed"
assert all(expected[key] is None for key in ("max_train_samples", "max_eval_samples", "max_audio_seconds")), "Scientific runs require unrestricted data"
assert not expected["freeze_backbone"] and expected["save_optimizer"], "Full-backbone/resumable scientific config required"
run = root / "runs" / run_id

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def check_metadata(path, require_complete=False):
    metadata = read(path / "metadata.json")
    assert metadata["run_id"] == run_id and metadata["stage"] == stage, "Checkpoint belongs to another run/stage"
    assert metadata["purpose"] == "scientific_training" and metadata["effective_config"] == expected, "Checkpoint config differs"
    for filename in ("model.safetensors", "config.json", "vocab.json"):
        assert (path / filename).is_file(), f"Checkpoint is incomplete: {path / filename}"
    if require_complete:
        completed = metadata.get("completed_training", {})
        assert completed.get("status") == "completed" and completed.get("optimizer_steps") == expected["max_steps"], "Checkpoint lacks full scientific completion"
    return metadata

if not run.exists():
    print(json.dumps({"action": "new", "max_steps": expected["max_steps"]}))
    raise SystemExit(0)
assert (run / "config.json").is_file(), f"Existing run has no config; preserve it and choose another experiment: {run}"
actual = read(run / "config.json")
differences = sorted(key for key in set(expected) | set(actual) if expected.get(key) != actual.get(key))
assert actual == expected, f"Existing run config differs in {differences}: {run}"
if stage in {"B", "C"}:
    best_a = root / "runs" / (run_id.rsplit(f"-{stage}-fold", 1)[0] + "-A") / "best_checkpoint"
    initialization = read(run / "initialization.json")
    initial_path = Path(initialization["checkpoint"])
    if not initial_path.is_absolute():
        initial_path = root / initial_path
    assert initial_path.resolve() == best_a.resolve(), "B/C must share this experiment's best A checkpoint"
    digest = hashlib.sha256()
    with (best_a / "model.safetensors").open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    assert initialization["weight_sha256"] == digest.hexdigest(), "B/C initialization weights differ from best A"
completed_path = run / "completed.json"
if completed_path.is_file():
    completed = read(completed_path)
    assert completed.get("status") == "completed" and completed.get("purpose") == "scientific_training", "Run is not a completed scientific run"
    assert completed.get("optimizer_steps") == expected["max_steps"], "Completed run has a different step budget"
    check_metadata(run / "best_checkpoint", require_complete=True)
    check_metadata(run / "checkpoint", require_complete=True)
    print(json.dumps({"action": "skip_completed", "max_steps": expected["max_steps"]}))
    raise SystemExit(0)
assert allow_resume == "1", f"Incomplete run exists: {run}. Use -ResumeExisting only to continue its checkpoint."
checkpoint = run / "checkpoint"
if not checkpoint.exists():
    checkpoint = run / "checkpoint.previous"
assert (checkpoint / "training_state.pt").is_file(), f"No resumable optimizer/RNG checkpoint: {run}; existing run preserved, no restart attempted."
metadata = check_metadata(checkpoint)
assert 0 < metadata["step"] < expected["max_steps"], "Checkpoint is not a resumable intermediate step"
assert (run / "initialization.json").is_file(), "Existing run initialization record is missing"
best_metadata = run / "best_checkpoint/metadata.json"
if best_metadata.is_file():
    assert read(best_metadata)["step"] <= metadata["step"], "Best checkpoint is newer than saved optimizer state"
print(json.dumps({"action": "resume", "checkpoint": str(checkpoint), "step": metadata["step"], "max_steps": expected["max_steps"]}))
'@

function Get-RunPlan {
    param([string]$StageName)
    $resumeFlag = if ($ResumeExisting) { '1' } else { '0' }
    $result = & $VenvPython -X utf8 -c $RunProbe $TaskRoot $StageName $RunIds[$StageName] "$Seed" "$Fold" $resumeFlag
    if ($LASTEXITCODE -ne 0) { throw "Scientific run verification failed for stage $StageName. Existing runs were preserved." }
    return ($result | ConvertFrom-Json -AsHashtable)
}

function Get-StageArguments {
    param([string]$StageName, [hashtable]$Plan)
    $parameters = @{ Stage = $StageName; RunId = $RunIds[$StageName]; Seed = $Seed }
    if ($StageName -ne 'A') { $parameters.Fold = $Fold }
    if ($Plan.action -eq 'resume') { $parameters.Resume = $Plan.checkpoint }
    elseif ($StageName -ne 'A') { $parameters.InitCheckpoint = $BestA }
    return $parameters
}

if ($DryRun) {
    Write-Host "Sequence: A (2000 steps), B (400 steps), C (400 steps); seed=$Seed fold=$Fold."
    Write-Host "State: $StatePath"
    Write-Host "Redirect stdout: $(Join-Path $StateDir 'stdout.log')"
    Write-Host "Redirect stderr: $(Join-Path $StateDir 'stderr.log')"
    Write-Host "B/C share completed best A: $BestA"
    foreach ($stageName in @('A', 'B', 'C')) {
        $arguments = Get-StageArguments $stageName @{ action = 'new' }
        & $Launcher @arguments -DryRun
    }
    Write-Host 'No training, state writes, or locks were started.'
    return
}

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) { throw 'Existing .venv is required. Run setup_local_training.ps1 first.' }
New-Item -ItemType Directory -Path $TrainingRoot -Force | Out-Null
New-Item -ItemType Directory -Path $StateDir -Force | Out-Null
$LockPath = Join-Path $TrainingRoot "$Experiment.lock"
$sequenceLock = $null
$state = $null

function Save-SequenceState {
    $state.updated_at = [DateTimeOffset]::UtcNow.ToString('o')
    $temporary = "$StatePath.$PID.tmp"
    [IO.File]::WriteAllText($temporary, ($state | ConvertTo-Json -Depth 16) + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
    [IO.File]::Move($temporary, $StatePath, $true)
}

try {
    try {
        $sequenceLock = [IO.File]::Open($LockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    }
    catch { throw "Another sequence for this experiment holds the lock, or it cannot be opened: $LockPath" }
    $lockBytes = [Text.Encoding]::UTF8.GetBytes("pid=$PID experiment=$Experiment fold=$Fold`n")
    $sequenceLock.SetLength(0)
    $sequenceLock.Write($lockBytes, 0, $lockBytes.Length)
    $sequenceLock.Flush()
    if (Test-Path -LiteralPath $StatePath -PathType Leaf) {
        $oldState = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json -AsHashtable
        if ($oldState.experiment -ne $Experiment -or $oldState.seed -ne $Seed -or $oldState.fold -ne $Fold) {
            throw 'StateDir belongs to another experiment, seed, or fold. Use its original parameters or choose another StateDir.'
        }
        $archivePath = Join-Path $StateDir ("state.before-" + [DateTimeOffset]::UtcNow.ToString('yyyyMMddTHHmmssfffffff') + '.json')
        Copy-Item -LiteralPath $StatePath -Destination $archivePath
    }
    $state = [ordered]@{
        schema_version = 1; experiment = $Experiment; seed = $Seed; fold = $Fold
        purpose = 'scientific_training'; device = 'cuda'; pid = $PID
        status = 'running'; phase = 'validating'; current_stage = $null; run_ids = $RunIds
        stdout_log = (Join-Path $StateDir 'stdout.log'); stderr_log = (Join-Path $StateDir 'stderr.log')
        started_at_utc = [DateTimeOffset]::UtcNow.ToString('o'); updated_at = $null
        stages = [ordered]@{}
    }
    foreach ($stageName in @('A', 'B', 'C')) {
        $state.stages[$stageName] = [ordered]@{
            status = 'pending'; run_id = $RunIds[$stageName]
            config = "configs/$($ConfigNames[$stageName])"
            run_dir = (Join-Path $TaskRoot "runs/$($RunIds[$stageName])")
            max_steps = $(if ($stageName -eq 'A') { 2000 } else { 400 })
            initialization_checkpoint = $(if ($stageName -eq 'A') { $null } else { $BestA })
        }
    }
    Save-SequenceState
    # Verify every existing run before starting any new optimizer updates.
    $plans = @{}
    foreach ($stageName in @('A', 'B', 'C')) { $plans[$stageName] = Get-RunPlan $stageName }
    foreach ($stageName in @('A', 'B', 'C')) {
        $state.current_stage = $stageName
        $plan = Get-RunPlan $stageName
        if ($plan.action -eq 'skip_completed') {
            $state.stages[$stageName].status = 'completed'
            $state.stages[$stageName].reused_existing = $true
            Save-SequenceState
            Write-Host "Verified completed scientific stage $stageName; reusing $($RunIds[$stageName])."
            continue
        }
        if ($stageName -ne 'A') {
            $sourcePlan = Get-RunPlan 'A'
            if ($sourcePlan.action -ne 'skip_completed') { throw 'B/C require the completed scientific best stage-A checkpoint.' }
        }
        $state.status = 'running'
        $state.phase = 'training'
        $state.stages[$stageName].status = 'running'
        $state.stages[$stageName].action = $plan.action
        $state.stages[$stageName].started_at = [DateTimeOffset]::UtcNow.ToString('o')
        if ($plan.action -eq 'resume') { $state.stages[$stageName].resume_checkpoint = $plan.checkpoint }
        Save-SequenceState
        Write-Host "Sequence PID $PID starting scientific stage $stageName ($($RunIds[$stageName]))."
        $arguments = Get-StageArguments $stageName $plan
        & $Launcher @arguments
        $finished = Get-RunPlan $stageName
        if ($finished.action -ne 'skip_completed') { throw "Stage $stageName exited without verified full scientific completion." }
        $state.stages[$stageName].status = 'completed'
        $state.stages[$stageName].completed_at = [DateTimeOffset]::UtcNow.ToString('o')
        Save-SequenceState
    }
    $state.status = 'completed'
    $state.phase = 'completed'
    $state.current_stage = $null
    $state.completed_at = [DateTimeOffset]::UtcNow.ToString('o')
    Save-SequenceState
    Write-Host "Scientific A/B/C sequence completed. State: $StatePath"
}
catch {
    if ($null -ne $state) {
        $state.status = 'failed'
        $state.error = $_.Exception.Message
        if ($state.current_stage) {
            $state.stages[$state.current_stage].status = 'failed'
            $state.stages[$state.current_stage].error = $_.Exception.Message
        }
        Save-SequenceState
    }
    throw
}
finally {
    if ($null -ne $sequenceLock) { $sequenceLock.Dispose() }
}
