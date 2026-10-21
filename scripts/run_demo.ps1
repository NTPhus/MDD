<#
.SYNOPSIS
Run the local pronunciation demo with CPU inference alongside CUDA training.
.EXAMPLE
pwsh -File scripts/run_demo.ps1
#>
[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)][int]$Port = 8765,
    [ValidateRange(1, 4)][int]$Threads = 2,
    [switch]$Lan,
    [string[]]$LanAddress = @(),
    [switch]$DryRun,
    [switch]$Help
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$TaskRoot = Split-Path -Parent $PSScriptRoot
if ($Help) {
    Write-Host @'
Usage: pwsh -File scripts/run_demo.ps1 [-Port 8765] [-Threads 2] [-Lan [-LanAddress 10.10.63.214]] [-DryRun]

Open the printed http://127.0.0.1 URL in your browser. Inference uses CPU;
the training process can continue on CUDA. Local MDD needs no API key or Node.
LLM guidance and ai33 voices use server-side keys and voice IDs from .env.
Stop this server with Ctrl+C. Checkpoints are discovered from runs/.
LAN mode allows only the selected RFC1918 IPv4 addresses. Configure Windows
Firewall with setup_demo_lan.ps1. HTTP LAN supports uploads/samples; mic needs HTTPS.
'@
    return
}
$DemoPython = Join-Path $TaskRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $DemoPython -PathType Leaf)) {
    throw 'Missing .venv. Run: pwsh -File scripts/setup_local_training.ps1'
}
if ($LanAddress.Count -and -not $Lan) { throw '-LanAddress requires -Lan.' }
if ($Lan) {
    if (-not $LanAddress.Count) {
        $LanAddress = @(Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway } | ForEach-Object { $_.IPv4Address.IPAddress } | Where-Object {
            $lanCandidate = $null
            if ([System.Net.IPAddress]::TryParse($_, [ref]$lanCandidate) -and $lanCandidate.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork) {
                $lanBytes = $lanCandidate.GetAddressBytes()
                $lanBytes[0] -eq 10 -or ($lanBytes[0] -eq 172 -and $lanBytes[1] -ge 16 -and $lanBytes[1] -le 31) -or ($lanBytes[0] -eq 192 -and $lanBytes[1] -eq 168)
            }
        } | Sort-Object -Unique)
    }
    if (-not $LanAddress.Count) { throw 'No LAN IPv4 with a default gateway found; pass -LanAddress explicitly.' }
    foreach ($lanIp in $LanAddress) {
        $lanParsed = $null
        if (-not [System.Net.IPAddress]::TryParse($lanIp, [ref]$lanParsed) -or $lanParsed.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork -or $lanParsed.IPAddressToString -ne $lanIp) {
            throw "Invalid LAN IPv4: $lanIp"
        }
        $lanBytes = $lanParsed.GetAddressBytes()
        if (-not ($lanBytes[0] -eq 10 -or ($lanBytes[0] -eq 172 -and $lanBytes[1] -ge 16 -and $lanBytes[1] -le 31) -or ($lanBytes[0] -eq 192 -and $lanBytes[1] -eq 168))) {
            throw "LAN IPv4 must be in 10/8, 172.16/12 or 192.168/16: $lanIp"
        }
    }
}
$DemoArguments = @('-X', 'utf8', '-u', (Join-Path $TaskRoot 'scripts/serve_demo.py'), '--port', "$Port", '--threads', "$Threads")
if ($Lan) {
    $DemoArguments += '--lan'
    foreach ($lanIp in $LanAddress) { $DemoArguments += @('--allow-host', $lanIp) }
}
if ($DryRun) {
    @{ python = $DemoPython; arguments = $DemoArguments; lan_addresses = @($LanAddress) } | ConvertTo-Json -Depth 4
    return
}
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:OMP_NUM_THREADS = "$Threads"
$env:MKL_NUM_THREADS = "$Threads"
& $DemoPython @DemoArguments
if ($LASTEXITCODE -ne 0) { throw "Demo server exited with code $LASTEXITCODE." }
