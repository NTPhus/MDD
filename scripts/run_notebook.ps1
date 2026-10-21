<#
.SYNOPSIS
Open the project demo notebook with the workspace .venv and localhost JupyterLab.
.EXAMPLE
pwsh -File scripts/run_notebook.ps1
.EXAMPLE
pwsh -File scripts/run_notebook.ps1 -Port 8890 -OpenBrowser
#>
[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)][int]$Port = 8888,
    [switch]$OpenBrowser,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$TaskRoot = Split-Path -Parent $PSScriptRoot
$VenvRoot = Join-Path $TaskRoot '.venv'
$VenvPython = Join-Path $VenvRoot 'Scripts/python.exe'
$NotebookPath = Join-Path $TaskRoot 'notebooks/mdd_local_demo_comparison.ipynb'
$KernelPath = Join-Path $VenvRoot 'share/jupyter/kernels/mdd-local/kernel.json'

if ($DryRun) {
    Write-Host "Python: $VenvPython"
    Write-Host "Notebook: $NotebookPath"
    Write-Host "Server: http://127.0.0.1:$Port (token authentication enabled; see startup URL)"
    Write-Host "Browser: $([bool]$OpenBrowser); kernel: MDD (.venv) [mdd-local]"
    Write-Host 'Command: python -m jupyterlab --ServerApp.ip=127.0.0.1 --ServerApp.port=<Port> --ServerApp.port_retries=0 --ServerApp.root_dir=<workspace> --ServerApp.default_url=/lab/tree/notebooks/mdd_local_demo_comparison.ipynb'
    Write-Host 'Jupyter config, data, runtime, and IPython directories stay inside .venv.'
    return
}

foreach ($requiredPath in @($VenvPython, $NotebookPath, $KernelPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file is missing: $requiredPath. Run pwsh -File scripts/setup_notebook.ps1 first and ensure the demo notebook exists."
    }
}
& $VenvPython -X utf8 -c 'import jupyterlab, ipykernel; import json, pathlib, sys; spec = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")); assert pathlib.Path(spec["argv"][0]).resolve() == pathlib.Path(sys.executable).resolve(), "Kernel points to another Python; rerun setup_notebook.ps1"' $KernelPath
if ($LASTEXITCODE -ne 0) { throw 'Notebook tools or the project kernel are not ready. Run pwsh -File scripts/setup_notebook.ps1.' }

$NotebookEnvironment = @{
    JUPYTER_CONFIG_DIR = (Join-Path $VenvRoot '.jupyter/config')
    JUPYTER_DATA_DIR = (Join-Path $VenvRoot 'share/jupyter')
    JUPYTER_RUNTIME_DIR = (Join-Path $VenvRoot '.jupyter/runtime')
    IPYTHONDIR = (Join-Path $VenvRoot '.ipython')
    PYTHONUTF8 = '1'
    PYTHONIOENCODING = 'utf-8'
}
$SavedEnvironment = @{}
foreach ($name in $NotebookEnvironment.Keys) {
    $SavedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    if ($name -notin @('PYTHONUTF8', 'PYTHONIOENCODING')) {
        New-Item -ItemType Directory -Path $NotebookEnvironment[$name] -Force | Out-Null
    }
    [Environment]::SetEnvironmentVariable($name, $NotebookEnvironment[$name], 'Process')
}
$SavedEnvironment['JUPYTER_PATH'] = [Environment]::GetEnvironmentVariable('JUPYTER_PATH', 'Process')
$kernelSearchPath = Join-Path $VenvRoot 'share/jupyter'
if ($SavedEnvironment['JUPYTER_PATH']) {
    $kernelSearchPath += [IO.Path]::PathSeparator + $SavedEnvironment['JUPYTER_PATH']
}
[Environment]::SetEnvironmentVariable('JUPYTER_PATH', $kernelSearchPath, 'Process')

Push-Location -LiteralPath $TaskRoot
try {
    $arguments = @(
        '-m', 'jupyterlab',
        '--ServerApp.ip=127.0.0.1',
        "--ServerApp.port=$Port",
        '--ServerApp.port_retries=0',
        "--ServerApp.root_dir=$TaskRoot",
        '--ServerApp.default_url=/lab/tree/notebooks/mdd_local_demo_comparison.ipynb',
        "--ServerApp.open_browser=$([bool]$OpenBrowser)"
    )
    Write-Host 'Select kernel MDD (.venv). Use the local URL printed by Jupyter; press Ctrl+C to stop.'
    & $VenvPython @arguments
    if ($LASTEXITCODE -ne 0) { throw "JupyterLab exited with code $LASTEXITCODE. If the port is occupied, use -Port 8890." }
}
finally {
    Pop-Location
    foreach ($name in $SavedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $SavedEnvironment[$name], 'Process')
    }
}
