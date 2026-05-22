# Run this script as Administrator (right-click PowerShell → "Run as administrator")
# It registers all 3 MLB scheduled tasks from the XML files in this folder.
#
# Usage:
#   cd C:\Users\josh\Git\SuperNovaBets\scripts
#   powershell -ExecutionPolicy Bypass -File install_tasks.ps1

$taskFolder = "SuperNovaBets"
$xmlDir     = "$PSScriptRoot\tasks"

# Create task subfolder if it doesn't exist
$schedService = New-Object -ComObject Schedule.Service
$schedService.Connect()
$rootFolder = $schedService.GetFolder("\")
try {
    $rootFolder.GetFolder($taskFolder) | Out-Null
} catch {
    $rootFolder.CreateFolder($taskFolder) | Out-Null
    Write-Host "Created task folder: \$taskFolder"
}

$tasks = @(
    @{ Name = "MLB-Morning";  File = "MLB-Morning.xml"  },
    @{ Name = "MLB-PreGame";  File = "MLB-PreGame.xml"  },
    @{ Name = "MLB-Close";    File = "MLB-Close.xml"    }
)

foreach ($t in $tasks) {
    $xmlPath  = Join-Path $xmlDir $t.File
    $taskName = "$taskFolder\$($t.Name)"
    $xml      = [System.IO.File]::ReadAllText($xmlPath, [System.Text.Encoding]::Unicode)

    try {
        Register-ScheduledTask -TaskName $taskName -Xml $xml -Force -ErrorAction Stop
        Write-Host "Registered: $taskName"
    } catch {
        Write-Error "Failed to register $taskName`: $_"
    }
}

Write-Host ""
Write-Host "Done. Verify in Task Scheduler under \SuperNovaBets\"
