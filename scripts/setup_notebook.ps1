<#
.SYNOPSIS
Add notebook tools and a workspace-local kernel to the existing .venv.
.EXAMPLE
pwsh -File scripts/setup_notebook.ps1
.EXAMPLE
pwsh -File scripts/setup_notebook.ps1 -Offline
#>
[CmdletBinding()]
param(
    [switch]$Offline,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$TaskRoot = Split-Path -Parent $PSScriptRoot
$VenvRoot = Join-Path $TaskRoot '.venv'
$VenvPython = Join-Path $VenvRoot 'Scripts/python.exe'
$RequirementsPath = Join-Path $TaskRoot 'requirements/notebook.txt'
$KernelPath = Join-Path $VenvRoot 'share/jupyter/kernels/mdd-local/kernel.json'

function Invoke-NotebookPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $VenvPython -X utf8 @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Notebook command failed (exit $LASTEXITCODE): $($Arguments -join ' ')"
    }
}

$RequirementsProbe = @'
from importlib import metadata
from pathlib import Path
import sys
try:
    from packaging.requirements import Requirement
except ImportError:
    raise SystemExit("Missing packaging in .venv; repair pip before notebook setup.")

problems = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    text = line.split("#", 1)[0].strip()
    if not text:
        continue
    requirement = Requirement(text)
    if requirement.marker and not requirement.marker.evaluate():
        continue
    try:
        installed = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        problems.append(f"{requirement}: not installed")
        continue
    if installed not in requirement.specifier:
        problems.append(f"{requirement}: installed {installed}")
if problems:
    print("Notebook requirements need attention:\n  " + "\n  ".join(problems))
    raise SystemExit(1)
print("All notebook requirements are installed within the supported version ranges.")
'@

$KernelProbe = @'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Workspace kernel is missing: {path}")
spec = json.loads(path.read_text(encoding="utf-8"))
if Path(spec["argv"][0]).resolve() != Path(sys.executable).resolve():
    raise SystemExit("Workspace kernel points to another Python; rerun setup_notebook.ps1.")
if spec.get("display_name") != "MDD (.venv)":
    raise SystemExit("Workspace kernel has an unexpected display name.")
print(f"Kernel ready: MDD (.venv) [mdd-local], Python {sys.executable}")
'@

if ($DryRun) {
    Write-Host "Reuse environment: $VenvPython"
    if ($Offline) {
        Write-Host 'Offline: verify installed requirements/notebook.txt; no downloads.'
        Write-Host 'Verify dependency metadata: python -m pip install --no-index --no-deps --dry-run -r requirements/notebook.txt'
    }
    else {
        Write-Host 'Install missing/out-of-range tools: python -m pip install -r requirements/notebook.txt'
        Write-Host 'Online install temporarily overrides inherited PIP_NO_INDEX; the original value is restored.'
    }
    Write-Host 'Check: python -m pip check'
    Write-Host "Register local kernel: python -m ipykernel install --prefix `"$VenvRoot`" --name mdd-local --display-name `"MDD (.venv)`""
    Write-Host "Verify kernel: $KernelPath"
    Write-Host 'Next: pwsh -File scripts/run_notebook.ps1'
    return
}

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "Existing training environment is required: $VenvPython. Create it with scripts/setup_local_training.ps1 first."
}

$SetupEnvironment = @{
    TEMP = (Join-Path $VenvRoot '.tmp')
    TMP = (Join-Path $VenvRoot '.tmp')
    IPYTHONDIR = (Join-Path $VenvRoot '.ipython')
    JUPYTER_CONFIG_DIR = (Join-Path $VenvRoot '.jupyter/config')
    JUPYTER_DATA_DIR = (Join-Path $VenvRoot 'share/jupyter')
}
$SavedEnvironment = @{}
foreach ($name in $SetupEnvironment.Keys) {
    $SavedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    New-Item -ItemType Directory -Path $SetupEnvironment[$name] -Force | Out-Null
    [Environment]::SetEnvironmentVariable($name, $SetupEnvironment[$name], 'Process')
}

Push-Location -LiteralPath $TaskRoot
try {
    & $VenvPython -X utf8 -c $RequirementsProbe $RequirementsPath
    $requirementsMatch = $LASTEXITCODE -eq 0
    if (-not $requirementsMatch) {
        if ($Offline) {
            throw 'Offline verification failed: install the missing/supported notebook tools by running pwsh -File scripts/setup_notebook.ps1 with network access, then retry -Offline.'
        }
        $savedPipNoIndex = [Environment]::GetEnvironmentVariable('PIP_NO_INDEX', 'Process')
        try {
            [Environment]::SetEnvironmentVariable('PIP_NO_INDEX', '0', 'Process')
            Invoke-NotebookPython -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '--retries', '2', '--timeout', '60', '--index-url', 'https://pypi.org/simple', '-r', $RequirementsPath)
        }
        finally {
            [Environment]::SetEnvironmentVariable('PIP_NO_INDEX', $savedPipNoIndex, 'Process')
        }
        Invoke-NotebookPython -Arguments @('-c', $RequirementsProbe, $RequirementsPath)
    }
    elseif ($Offline) {
        Invoke-NotebookPython -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-index', '--no-deps', '--dry-run', '-r', $RequirementsPath)
    }
    Invoke-NotebookPython -Arguments @('-m', 'pip', 'check')
    Invoke-NotebookPython -Arguments @('-m', 'ipykernel', 'install', '--prefix', $VenvRoot, '--name', 'mdd-local', '--display-name', 'MDD (.venv)')
    Invoke-NotebookPython -Arguments @('-c', $KernelProbe, $KernelPath)
    Write-Host 'Notebook setup is ready. Select kernel MDD (.venv).'
    Write-Host 'Open: pwsh -File scripts/run_notebook.ps1'
}
finally {
    Pop-Location
    foreach ($name in $SavedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $SavedEnvironment[$name], 'Process')
    }
}
