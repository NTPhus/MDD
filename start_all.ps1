<#
.SYNOPSIS
Run or manage the entire MDD system (Backend + Cloudflare Tunnel) with one command.

.EXAMPLE
.\start_all.ps1
.\start_all.ps1 -Action Restart
.\start_all.ps1 -Action Stop
.\start_all.ps1 -Action Status
.\start_all.ps1 -LocalOnly
#>
[CmdletBinding()]
param(
    [ValidateSet('start', 'stop', 'restart', 'status')][string]$Action = 'start',
    [switch]$LocalOnly,
    [switch]$Daemon,
    [int]$Threads = 2,
    [switch]$Json
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$TaskRoot = $PSScriptRoot
$DemoPython = Join-Path $TaskRoot '.venv/Scripts/python.exe'
$RunScript = Join-Path $TaskRoot 'scripts/run_system.py'

if (-not (Test-Path -LiteralPath $DemoPython -PathType Leaf)) {
    throw 'Missing .venv virtual environment. Please run setup first.'
}

$Arguments = @($RunScript, $Action.ToLowerInvariant())
if ($LocalOnly) { $Arguments += '--local-only' }
if ($Daemon) { $Arguments += '--daemon' }
if ($Threads) { $Arguments += @('--threads', "$Threads") }
if ($Json) { $Arguments += '--json' }

& $DemoPython @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "System command exited with code $LASTEXITCODE."
}
