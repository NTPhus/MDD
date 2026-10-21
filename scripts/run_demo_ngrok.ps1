<#
.SYNOPSIS
Set up, start, inspect or stop an isolated HTTPS ngrok ingress for the running demo.
.EXAMPLE
pwsh -File scripts/run_demo_ngrok.ps1 -Action Start
#>
[CmdletBinding()]
param(
    [ValidateSet('Setup', 'Start', 'Status', 'Stop')][string]$Action = 'Start',
    [ValidateRange(1024, 65535)][int]$IngressPort = 8081,
    [ValidateRange(1024, 65535)][int]$ControlPort = 4040,
    [ValidateRange(1024, 65535)][int]$BackendPort = 8765
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$TaskRoot = Split-Path -Parent $PSScriptRoot
$DemoPython = Join-Path $TaskRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $DemoPython -PathType Leaf)) { throw 'Missing .venv; set up the local demo first.' }
$DemoArguments = @('-X', 'utf8', (Join-Path $TaskRoot 'scripts/deploy_demo_ngrok.py'), '--action', $Action.ToLowerInvariant(), '--ingress-port', "$IngressPort", '--control-port', "$ControlPort", '--backend-port', "$BackendPort")
& $DemoPython @DemoArguments
if ($LASTEXITCODE -ne 0) { throw "ngrok deployment exited with code $LASTEXITCODE." }
