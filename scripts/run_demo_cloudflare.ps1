<#
.SYNOPSIS
Start, inspect, or stop a temporary Cloudflare Quick Tunnel for the running MDD demo.
.EXAMPLE
pwsh -File scripts/run_demo_cloudflare.ps1 -Action Start
#>
[CmdletBinding()]
param([ValidateSet('Start', 'Status', 'Stop')][string]$Action = 'Start')
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$TaskRoot = Split-Path -Parent $PSScriptRoot
$DemoPython = Join-Path $TaskRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $DemoPython -PathType Leaf)) { throw 'Missing .venv; set up the local demo first.' }
& $DemoPython -X utf8 (Join-Path $TaskRoot 'scripts/deploy_demo_cloudflare.py') --action $Action.ToLowerInvariant()
if ($LASTEXITCODE -ne 0) { throw "Cloudflare deployment exited with code $LASTEXITCODE." }
