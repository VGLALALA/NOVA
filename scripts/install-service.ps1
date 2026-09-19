# Install NOVA as a Windows Scheduled Task (dashboard or worker).
# Usage:
#   .\scripts\install-service.ps1 dashboard
#   .\scripts\install-service.ps1 worker
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("dashboard", "worker")]
    [string]$Role
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Nova = Join-Path $Root ".venv\Scripts\nova.exe"
if (-not (Test-Path $Nova)) {
    throw "Missing $Nova — run .\scripts\setup.ps1 first."
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root ".nova") | Out-Null
$TaskName = "NOVA-$Role"
$Arg = if ($Role -eq "dashboard") { "dashboard" } else { "worker" }
$Action = New-ScheduledTaskAction -Execute $Nova -Argument $Arg -WorkingDirectory $Root
$Trigger = New-ScheduledTaskTrigger -AtLogOn
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Host "installed scheduled task $TaskName"
Write-Host "  Get-ScheduledTask $TaskName"
Write-Host "  Stop with: Stop-ScheduledTask -TaskName $TaskName"
