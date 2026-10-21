<#
.SYNOPSIS
Allow this workspace's Nginx on one LAN interface, subnet and optional clients.
.EXAMPLE
pwsh -File scripts/setup_demo_nginx_firewall.ps1 -LanAddress 10.10.63.214 -InterfaceAlias Wi-Fi -ProgramPath .tools/nginx/nginx.exe -AllowedClient 10.10.60.25 -DryRun
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$LanAddress,
    [Parameter(Mandatory = $true)][string]$InterfaceAlias,
    [Parameter(Mandatory = $true)][string]$ProgramPath,
    [ValidateSet('Domain', 'Private', 'Public')][string]$Profile = 'Public',
    [ValidateRange(1, 65535)][int]$Port = 80,
    [string[]]$AllowedClient = @(),
    [switch]$VerifyOnly,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$TaskRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$ResultDirectory = Join-Path $TaskRoot 'data/generated/demo/nginx'
[void](New-Item -ItemType Directory -Path $ResultDirectory -Force)
$ResultPath = Join-Path $ResultDirectory 'firewall-result.json'
$NginxRuleName = $null

function Get-PrivateIPv4([string]$Address, [string]$Label) {
    $parsedAddress = $null
    if (-not [Net.IPAddress]::TryParse($Address, [ref]$parsedAddress) -or
        $parsedAddress.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork -or
        $parsedAddress.IPAddressToString -ne $Address) {
        throw "$Label must be a canonical IPv4 address."
    }
    $addressBytes = $parsedAddress.GetAddressBytes()
    if (-not ($addressBytes[0] -eq 10 -or
        ($addressBytes[0] -eq 172 -and $addressBytes[1] -ge 16 -and $addressBytes[1] -le 31) -or
        ($addressBytes[0] -eq 192 -and $addressBytes[1] -eq 168))) {
        throw "$Label must be an RFC1918 IPv4 address."
    }
    return $parsedAddress
}

try {
    if ($DryRun -and $VerifyOnly) { throw 'Choose DryRun or VerifyOnly, not both.' }
    if ([string]::IsNullOrWhiteSpace($InterfaceAlias)) { throw 'InterfaceAlias is required.' }
    $nginxRoot = [IO.Path]::GetFullPath((Join-Path $TaskRoot '.tools/nginx'))
    $nginxProgram = (Resolve-Path -LiteralPath $ProgramPath).ProviderPath
    if (-not (Test-Path -LiteralPath $nginxProgram -PathType Leaf) -or
        [IO.Path]::GetFileName($nginxProgram) -ine 'nginx.exe' -or
        -not $nginxProgram.StartsWith($nginxRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'ProgramPath must be an existing nginx.exe inside this workspace .tools/nginx directory.'
    }
    # Reject junctions/symlinks so the apparent workspace path cannot escape it.
    $programAncestor = Get-Item -LiteralPath $nginxProgram -Force
    while ($programAncestor -and $programAncestor.FullName -ne $TaskRoot) {
        if (($programAncestor.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw 'The Nginx program path must not contain junctions or symbolic links.'
        }
        if ($programAncestor -is [IO.FileInfo]) { $programAncestor = $programAncestor.Directory }
        else { $programAncestor = $programAncestor.Parent }
    }
    $lanParsed = Get-PrivateIPv4 $LanAddress 'LanAddress'
    $lanBytes = $lanParsed.GetAddressBytes()
    $lanInterface = @(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias $InterfaceAlias -IPAddress $LanAddress)
    if ($lanInterface.Count -ne 1 -or [string]$lanInterface[0].InterfaceAlias -cne $InterfaceAlias) {
        throw 'Address must belong to the exact selected local interface.'
    }
    $lanProfile = @(Get-NetConnectionProfile -InterfaceIndex $lanInterface[0].InterfaceIndex)
    if ($lanProfile.Count -ne 1) { throw 'The interface must have one active connection profile.' }
    $expectedProfile = switch ([string]$lanProfile[0].NetworkCategory) {
        '0' { 'Public' }; 'Public' { 'Public' }
        '1' { 'Private' }; 'Private' { 'Private' }
        '2' { 'Domain' }; 'DomainAuthenticated' { 'Domain' }
        default { throw 'Unknown network profile.' }
    }
    if ($Profile -ne $expectedProfile) { throw "Profile must match the active network ($expectedProfile); the script does not change network category." }
    $lanPrefix = [int]$lanInterface[0].PrefixLength
    $maskBytes = [byte[]](0..3 | ForEach-Object {
        $bits = [Math]::Max(0, [Math]::Min(8, $lanPrefix - $_ * 8))
        [byte](256 - [Math]::Pow(2, 8 - $bits))
    })
    $networkBytes = [byte[]](0..3 | ForEach-Object { [byte]($lanBytes[$_] -band $maskBytes[$_]) })
    $lastBytes = [byte[]](0..3 | ForEach-Object { [byte]($networkBytes[$_] -bor (255 - $maskBytes[$_])) })
    $networkAddress = ([Net.IPAddress]::new($networkBytes)).IPAddressToString
    $lastAddress = ([Net.IPAddress]::new($lastBytes)).IPAddressToString
    $null = Get-PrivateIPv4 $networkAddress 'Interface subnet'
    $null = Get-PrivateIPv4 $lastAddress 'Interface subnet'
    $lanSubnet = "$networkAddress/$lanPrefix"
    $subnetWithMask = $networkAddress + '/' + ([Net.IPAddress]::new($maskBytes)).IPAddressToString
    $remoteScopes = @($lanSubnet)
    $allowedClients = @($AllowedClient | Sort-Object -Unique)
    foreach ($clientAddress in $allowedClients) {
        $clientBytes = (Get-PrivateIPv4 $clientAddress 'AllowedClient').GetAddressBytes()
        $covered = $true
        for ($byteIndex = 0; $byteIndex -lt 4; $byteIndex++) {
            if (($clientBytes[$byteIndex] -band $maskBytes[$byteIndex]) -ne $networkBytes[$byteIndex]) { $covered = $false }
        }
        if (-not $covered) { $remoteScopes += $clientAddress }
    }
    $rootHasher = [Security.Cryptography.SHA256]::Create()
    try {
        $digest = [BitConverter]::ToString($rootHasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($TaskRoot.ToLowerInvariant()))).Replace('-', '').Substring(0, 10).ToLowerInvariant()
    } finally { $rootHasher.Dispose() }
    $NginxRuleName = "MDD-Nginx-$digest-$Port"
    $settings = @{
        DisplayName = "MDD Nginx demo ($Port)"
        Direction = 'Inbound'; Action = 'Allow'; Enabled = 'True'
        Profile = $Profile; Protocol = 'TCP'; LocalPort = $Port
        LocalAddress = $LanAddress; RemoteAddress = $remoteScopes
        InterfaceAlias = [Management.Automation.WildcardPattern]::Escape($InterfaceAlias)
        Program = $nginxProgram; EdgeTraversalPolicy = 'Block'
    }
    $plan = @{
        status = 'planned'; rule_name = $NginxRuleName; interface = $InterfaceAlias
        address = $LanAddress; remote_subnet = $lanSubnet; allowed_clients = @($allowedClients)
        remote_addresses = @($remoteScopes); profile = $Profile; port = $Port
        program = $nginxProgram; url = "http://${LanAddress}:$Port/"
    }
    if ($DryRun) {
        $plan | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $ResultDirectory 'firewall-plan.json') -Encoding utf8
        $plan | ConvertTo-Json -Depth 4
        return
    }
    if (-not $VerifyOnly) {
        $principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
        if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
            throw 'Windows Administrator rights are required to add the firewall rule.'
        }
        $existing = @(Get-NetFirewallRule -PolicyStore PersistentStore -Name $NginxRuleName -ErrorAction SilentlyContinue)
        if ($existing.Count -gt 0) {
            if ($existing.Count -ne 1 -or [string]$existing[0].Group -ne 'MDD Nginx Demo') {
                throw 'The rule name belongs to another application; no change made.'
            }
            $updateSettings = $settings.Clone()
            [void]$updateSettings.Remove('DisplayName')
            Set-NetFirewallRule -PolicyStore PersistentStore -Name $NginxRuleName -NewDisplayName $settings.DisplayName @updateSettings | Out-Null
        } else {
            New-NetFirewallRule -PolicyStore PersistentStore -Name $NginxRuleName -Group 'MDD Nginx Demo' @settings | Out-Null
        }
    }
    $rule = @(Get-NetFirewallRule -PolicyStore ActiveStore -Name $NginxRuleName)
    if ($rule.Count -ne 1) { throw 'Expected exactly one active Nginx firewall rule.' }
    $addresses = $rule[0] | Get-NetFirewallAddressFilter
    $ports = $rule[0] | Get-NetFirewallPortFilter
    $application = $rule[0] | Get-NetFirewallApplicationFilter
    $interface = $rule[0] | Get-NetFirewallInterfaceFilter
    $actualScopes = @($addresses.RemoteAddress | ForEach-Object {
        if ($_ -eq $subnetWithMask) { $lanSubnet }
        elseif ($_ -like '*/255.255.255.255') { $_.Substring(0, $_.Length - '/255.255.255.255'.Length) }
        elseif ($_ -like '*/32') { $_.Substring(0, $_.Length - 3) }
        else { $_ }
    } | Sort-Object -Unique)
    $scopeDifference = @(Compare-Object -ReferenceObject @($remoteScopes | Sort-Object -Unique) -DifferenceObject $actualScopes)
    if ([string]$rule[0].Group -ne 'MDD Nginx Demo' -or [string]$rule[0].Enabled -ne 'True' -or
        [string]$rule[0].Action -ne 'Allow' -or [string]$rule[0].Direction -ne 'Inbound' -or
        [string]$rule[0].Profile -ne $Profile -or [string]$rule[0].EdgeTraversalPolicy -notin @('Block', '0') -or
        @($addresses.LocalAddress).Count -ne 1 -or @($addresses.LocalAddress) -notcontains $LanAddress -or
        $scopeDifference.Count -ne 0 -or @($ports.LocalPort).Count -ne 1 -or @($ports.LocalPort) -notcontains "$Port" -or
        [string]$ports.Protocol -notin @('TCP', '6') -or [string]$application.Program -ine $nginxProgram -or
        @($interface.InterfaceAlias).Count -ne 1 -or @($interface.InterfaceAlias) -notcontains $InterfaceAlias) {
        throw 'Firewall verification did not match the requested narrow Nginx scope.'
    }
    $plan.status = 'ready'
    $plan.verified_remote_address = @($addresses.RemoteAddress)
    $plan | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $ResultPath -Encoding utf8
    $plan | ConvertTo-Json -Depth 4
} catch {
    @{
        status = 'error'; rule_name = $NginxRuleName; allowed_clients = @($AllowedClient)
        verification_only = [bool]$VerifyOnly; message = $_.Exception.Message
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $ResultPath -Encoding utf8
    throw
}
