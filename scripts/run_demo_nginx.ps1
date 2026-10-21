<#
.SYNOPSIS
Set up, start, inspect or stop this workspace's Nginx reverse proxy and MDD backend.
.EXAMPLE
pwsh -File scripts/run_demo_nginx.ps1 -LanAddress 10.10.63.214
#>
[CmdletBinding()]
param(
    [ValidateSet('Setup', 'Start', 'Stop', 'Status')][string]$Action = 'Start',
    [string]$LanAddress,
    [ValidateRange(1, 65535)][int]$Port = 80,
    [ValidateRange(1024, 65535)][int]$BackendPort = 8765,
    [ValidateRange(1, 4)][int]$Threads = 2
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$TaskRoot = Split-Path -Parent $PSScriptRoot
$DemoPython = Join-Path $TaskRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $DemoPython -PathType Leaf)) { throw 'Missing .venv; run setup_local_training.ps1 first.' }
$DemoArguments = @('-X', 'utf8', (Join-Path $TaskRoot 'scripts/deploy_demo_nginx.py'), '--action', $Action.ToLowerInvariant(), '--port', "$Port", '--backend-port', "$BackendPort", '--threads', "$Threads")
if ($LanAddress) { $DemoArguments += @('--lan-address', $LanAddress) }
& $DemoPython @DemoArguments
if ($LASTEXITCODE -ne 0) { throw "Nginx deployment exited with code $LASTEXITCODE." }
