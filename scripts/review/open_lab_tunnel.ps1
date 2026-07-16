param(
    [int[]]$Ports = @(2719, 2720, 2721),
    [string]$User = "bowen",
    [string]$HostName = "valkyrie03.gavo.t.u-tokyo.ac.jp",
    [string]$RemoteHost = "127.0.0.1"
)

if (-not $Ports -or $Ports.Count -eq 0) {
    Write-Error "At least one port is required."
    exit 2
}

$uniquePorts = $Ports | Sort-Object -Unique

Write-Host "Opening SSH tunnel to $User@$HostName"
Write-Host ""
Write-Host "Open these URLs in your local browser after the tunnel starts:"
foreach ($port in $uniquePorts) {
    Write-Host "  http://127.0.0.1:$port"
}
Write-Host ""
Write-Host "Press Ctrl+C in this terminal to close the tunnel."
Write-Host ""

$sshArgs = @("-N")
foreach ($port in $uniquePorts) {
    $sshArgs += @("-L", "${port}:${RemoteHost}:${port}")
}
$sshArgs += "${User}@${HostName}"

ssh @sshArgs
