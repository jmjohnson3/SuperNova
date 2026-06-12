# Removes known legacy MLB tasks and registers the maintained SuperNovaBets set.
# The script requests Administrator elevation when needed.
#
# Usage:
#   cd C:\Users\josh\Git\SuperNovaBets\scripts
#   powershell -ExecutionPolicy Bypass -File install_tasks.ps1

param(
    [switch]$KeepLegacyTasks
)

$ErrorActionPreference = "Stop"

$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [System.Security.Principal.WindowsPrincipal]::new($identity)
$isAdministrator = $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdministrator) {
    $arguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$PSCommandPath`""
    )
    if ($KeepLegacyTasks) {
        $arguments += "-KeepLegacyTasks"
    }
    try {
        $process = Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList $arguments `
            -WindowStyle Hidden -Wait -PassThru
        exit $process.ExitCode
    } catch {
        Write-Error "Administrator elevation was not approved. No scheduled tasks were changed."
        exit 1
    }
}

$taskFolder = "SuperNovaBets"
$xmlDir     = "$PSScriptRoot\tasks"
$taskPath   = "\$taskFolder\"
$failures   = [System.Collections.Generic.List[string]]::new()

$legacyTasks = @(
    @{ Path = "\"; Name = "MLB" },
    @{ Path = "\"; Name = "mlb1" },
    @{ Path = "\"; Name = "mlb_5am" },
    @{ Path = "\"; Name = "MLB_7am" },
    @{ Path = "\"; Name = "SuperNovaBets_MLB_CloseOnly" },
    @{ Path = "\"; Name = "SuperNovaBets_MLB_Daily" },
    @{ Path = $taskPath; Name = "MLB-PreGame" }
)

if (-not $KeepLegacyTasks) {
    foreach ($legacy in $legacyTasks) {
        $existing = Get-ScheduledTask -TaskPath $legacy.Path -TaskName $legacy.Name -ErrorAction SilentlyContinue
        if (-not $existing) {
            continue
        }
        try {
            if ($existing.State -eq "Running") {
                Stop-ScheduledTask -TaskPath $legacy.Path -TaskName $legacy.Name
            }
            Unregister-ScheduledTask -TaskPath $legacy.Path -TaskName $legacy.Name -Confirm:$false
            Write-Host "Removed legacy task: $($legacy.Path)$($legacy.Name)"
        } catch {
            $message = "Could not remove legacy task $($legacy.Path)$($legacy.Name): $($_.Exception.Message)"
            $failures.Add($message)
            Write-Warning $message
        }
    }
}

# Create task subfolder if it doesn't exist
try {
    $schedService = New-Object -ComObject Schedule.Service
    $schedService.Connect()
    $rootFolder = $schedService.GetFolder("\")
    try {
        $rootFolder.GetFolder($taskFolder) | Out-Null
    } catch {
        $rootFolder.CreateFolder($taskFolder) | Out-Null
        Write-Host "Created task folder: \$taskFolder"
    }
} catch {
    $message = "Could not create or access task folder $taskPath`: $($_.Exception.Message)"
    $failures.Add($message)
    Write-Warning $message
}

$tasks = @(
    @{ Name = "MLB-Morning";          File = "MLB-Morning.xml"          },
    @{ Name = "MLB-PreGame-Day";      File = "MLB-PreGame.xml"          },
    @{ Name = "MLB-PreGame-Evening";  File = "MLB-PreGame-Evening.xml"  },
    @{ Name = "MLB-Close";            File = "MLB-Close.xml"            },
    @{ Name = "MLB-Training";         File = "MLB-Training.xml"         }
)

foreach ($t in $tasks) {
    $xmlPath  = Join-Path $xmlDir $t.File
    $xml      = [System.IO.File]::ReadAllText($xmlPath, [System.Text.Encoding]::UTF8)
    # Register-ScheduledTask receives a .NET string, so an explicit UTF-8
    # declaration makes Task Scheduler reject the otherwise valid XML.
    $xml      = $xml -replace '^\s*<\?xml[^?]*\?>\s*', ''

    try {
        Register-ScheduledTask -TaskPath $taskPath -TaskName $t.Name -Xml $xml -Force -ErrorAction Stop | Out-Null
        Write-Host "Registered: $taskPath$($t.Name)"
    } catch {
        $message = "Failed to register $taskPath$($t.Name): $($_.Exception.Message)"
        $failures.Add($message)
        Write-Warning $message
    }
}

Write-Host ""
if ($failures.Count -gt 0) {
    Write-Host "Completed with $($failures.Count) issue(s). Run this script from an Administrator PowerShell to finish cleanup."
    exit 1
}

Write-Host "Done. Verify in Task Scheduler under \SuperNovaBets\"
