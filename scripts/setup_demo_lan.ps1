<#
.SYNOPSIS
Allow the MDD demo on one IPv4 LAN interface and subnet in Windows Firewall.
.EXAMPLE
pwsh -File scripts/setup_demo_lan.ps1 -LanAddress 10.10.63.214 -InterfaceAlias Wi-Fi -Profile Public -DryRun
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$LanAddress,
    [Parameter(Mandatory = $true)][string]$InterfaceAlias,
    [ValidateSet('Domain', 'Private', 'Public')][string]$Profile = 'Private',
    [ValidateRange(1024, 65535)][int]$Port = 8765,
    [string[]]$AllowedClient = @(),
    [switch]$VerifyOnly,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$TaskRoot = Split-Path -Parent $PSScriptRoot
$ResultDirectory = Join-Path $TaskRoot 'data/generated/demo/lan'
[void](New-Item -ItemType Directory -Path $ResultDirectory -Force)
$ResultPath = Join-Path $ResultDirectory 'firewall-result.json'
$LanRuleName = $null
try {
    $lanParsed = $null
    if (-not [System.Net.IPAddress]::TryParse($LanAddress, [ref]$lanParsed) -or $lanParsed.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork -or $lanParsed.IPAddressToString -ne $LanAddress) {
        throw 'LanAddress must be a canonical IPv4 address.'
    }
    $lanBytes = $lanParsed.GetAddressBytes()
    if (-not ($lanBytes[0] -eq 10 -or ($lanBytes[0] -eq 172 -and $lanBytes[1] -ge 16 -and $lanBytes[1] -le 31) -or ($lanBytes[0] -eq 192 -and $lanBytes[1] -eq 168))) {
        throw 'LanAddress must be an RFC1918 IPv4 address.'
    }
    $lanInterface = @(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias $InterfaceAlias -IPAddress $LanAddress)
    if ($lanInterface.Count -ne 1) { throw 'Address must belong to the selected local interface.' }
    $lanProfile = @(Get-NetConnectionProfile -InterfaceIndex $lanInterface[0].InterfaceIndex)
    if ($lanProfile.Count -ne 1) { throw 'The selected interface has no single active connection profile.' }
    $lanExpectedProfile = switch ([string]$lanProfile[0].NetworkCategory) {
        '0' { 'Public' }; 'Public' { 'Public' }
        '1' { 'Private' }; 'Private' { 'Private' }
        '2' { 'Domain' }; 'DomainAuthenticated' { 'Domain' }
        default { throw 'Unknown network profile.' }
    }
    if ($Profile -ne $lanExpectedProfile) { throw "Profile must match the active network ($lanExpectedProfile); the script does not change network category." }
    $lanPrefix = [int]$lanInterface[0].PrefixLength
    $lanNetworkBytes = [byte[]](0..3 | ForEach-Object {
        $lanBits = [Math]::Max(0, [Math]::Min(8, $lanPrefix - $_ * 8))
        $lanMask = [int](256 - [Math]::Pow(2, 8 - $lanBits))
        [byte]($lanBytes[$_] -band $lanMask)
    })
    $lanSubnet = ([System.Net.IPAddress]::new($lanNetworkBytes)).IPAddressToString + "/$lanPrefix"
    $lanNetmaskBytes = [byte[]](0..3 | ForEach-Object {
        $lanBits = [Math]::Max(0, [Math]::Min(8, $lanPrefix - $_ * 8))
        [byte](256 - [Math]::Pow(2, 8 - $lanBits))
    })
    $lanSubnetWithMask = ([System.Net.IPAddress]::new($lanNetworkBytes)).IPAddressToString + '/' + ([System.Net.IPAddress]::new($lanNetmaskBytes)).IPAddressToString
    $lanRemoteScopes = @($lanSubnet)
    $lanAllowedClients = @($AllowedClient | Sort-Object -Unique)
    foreach ($clientAddress in $lanAllowedClients) {
        $clientParsed = $null
        if (-not [System.Net.IPAddress]::TryParse($clientAddress, [ref]$clientParsed) -or $clientParsed.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork -or $clientParsed.IPAddressToString -ne $clientAddress) {
            throw 'AllowedClient must contain canonical IPv4 addresses.'
        }
        $clientBytes = $clientParsed.GetAddressBytes()
        if (-not ($clientBytes[0] -eq 10 -or ($clientBytes[0] -eq 172 -and $clientBytes[1] -ge 16 -and $clientBytes[1] -le 31) -or ($clientBytes[0] -eq 192 -and $clientBytes[1] -eq 168))) {
            throw 'AllowedClient must contain RFC1918 IPv4 addresses.'
        }
        $clientAlreadyCovered = $true
        for ($clientByteIndex = 0; $clientByteIndex -lt 4; $clientByteIndex++) {
            if (($clientBytes[$clientByteIndex] -band $lanNetmaskBytes[$clientByteIndex]) -ne $lanNetworkBytes[$clientByteIndex]) { $clientAlreadyCovered = $false }
        }
        if (-not $clientAlreadyCovered) { $lanRemoteScopes += $clientAddress }
    }
    $lanPython = Join-Path $TaskRoot '.venv/Scripts/python.exe'
    $lanProgram = (& $lanPython -X utf8 -c 'import sys; print(sys._base_executable)').Trim()
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $lanProgram -PathType Leaf)) { throw 'Cannot resolve the Python executable that owns the listening socket.' }
    $lanHash = [System.Security.Cryptography.SHA256]::Create()
    try { $lanDigest = [BitConverter]::ToString($lanHash.ComputeHash([Text.Encoding]::UTF8.GetBytes($TaskRoot.ToLowerInvariant()))).Replace('-', '').Substring(0, 10).ToLowerInvariant() }
    finally { $lanHash.Dispose() }
    $LanRuleName = "MDD-Demo-LAN-$lanDigest-$Port"
    $lanSettings = @{
        DisplayName = "MDD demo LAN ($Port)"
        Direction = 'Inbound'
        Action = 'Allow'
        Enabled = 'True'
        Profile = $Profile
        Protocol = 'TCP'
        LocalPort = $Port
        LocalAddress = $LanAddress
        RemoteAddress = $lanRemoteScopes
        InterfaceAlias = [System.Management.Automation.WildcardPattern]::Escape($InterfaceAlias)
        Program = $lanProgram
        EdgeTraversalPolicy = 'Block'
    }
    $lanPlan = @{ status = 'planned'; rule_name = $LanRuleName; interface = $InterfaceAlias; address = $LanAddress; remote_subnet = $lanSubnet; allowed_clients = @($lanAllowedClients); remote_addresses = @($lanRemoteScopes); profile = $Profile; port = $Port; program = $lanProgram; url = "http://${LanAddress}:$Port/" }
    if ($DryRun) {
        $lanPlan | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $ResultDirectory 'firewall-plan.json') -Encoding utf8
        $lanPlan | ConvertTo-Json -Depth 4
        return
    }
    if (-not $VerifyOnly) {
        $lanIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $lanPrincipal = [Security.Principal.WindowsPrincipal]::new($lanIdentity)
        if (-not $lanPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Windows Administrator rights are required to add the firewall rule.' }
        $lanExisting = Get-NetFirewallRule -PolicyStore PersistentStore -Name $LanRuleName -ErrorAction SilentlyContinue
        if ($lanExisting) {
            if ([string]$lanExisting.Group -ne 'MDD Local Demo') { throw 'The rule name belongs to another application; no change made.' }
            $lanUpdateSettings = $lanSettings.Clone()
            [void]$lanUpdateSettings.Remove('DisplayName')
            Set-NetFirewallRule -PolicyStore PersistentStore -Name $LanRuleName -NewDisplayName $lanSettings.DisplayName @lanUpdateSettings | Out-Null
        } else {
            New-NetFirewallRule -PolicyStore PersistentStore -Name $LanRuleName -Group 'MDD Local Demo' @lanSettings | Out-Null
        }
    }
    $lanRule = Get-NetFirewallRule -PolicyStore ActiveStore -Name $LanRuleName
    $lanAddresses = $lanRule | Get-NetFirewallAddressFilter
    $lanPorts = $lanRule | Get-NetFirewallPortFilter
    $lanApp = $lanRule | Get-NetFirewallApplicationFilter
    $lanNic = $lanRule | Get-NetFirewallInterfaceFilter
    $lanActualRemoteScopes = @($lanAddresses.RemoteAddress | ForEach-Object {
        if ($_ -eq $lanSubnetWithMask) { $lanSubnet }
        elseif ($_ -like '*/255.255.255.255') { $_.Substring(0, $_.Length - '/255.255.255.255'.Length) }
        elseif ($_ -like '*/32') { $_.Substring(0, $_.Length - '/32'.Length) }
        else { $_ }
    } | Sort-Object -Unique)
    $lanScopeDifference = @(Compare-Object -ReferenceObject @($lanRemoteScopes | Sort-Object -Unique) -DifferenceObject $lanActualRemoteScopes)
    if ([string]$lanRule.Group -ne 'MDD Local Demo' -or [string]$lanRule.Enabled -ne 'True' -or [string]$lanRule.Action -ne 'Allow' -or [string]$lanRule.Direction -ne 'Inbound' -or [string]$lanRule.Profile -ne $Profile -or
        @($lanAddresses.LocalAddress).Count -ne 1 -or @($lanAddresses.LocalAddress) -notcontains $LanAddress -or
        $lanScopeDifference.Count -ne 0 -or
        @($lanPorts.LocalPort) -notcontains "$Port" -or [string]$lanPorts.Protocol -notin @('TCP', '6') -or [string]$lanApp.Program -ne $lanProgram -or @($lanNic.InterfaceAlias) -notcontains $InterfaceAlias) {
        throw 'Firewall rule verification did not match the requested narrow LAN scope.'
    }
    $lanPlan.status = 'ready'
    $lanPlan.verified_remote_address = @($lanAddresses.RemoteAddress)
    $lanPlan | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $ResultPath -Encoding utf8
    $lanPlan | ConvertTo-Json -Depth 4
} catch {
    @{status = 'error'; rule_name = $LanRuleName; allowed_clients = @($AllowedClient); verification_only = [bool]$VerifyOnly; message = $_.Exception.Message} | ConvertTo-Json | Set-Content -LiteralPath $ResultPath -Encoding utf8
    throw
}
