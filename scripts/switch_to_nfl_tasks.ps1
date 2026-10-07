param(
    [switch]$NoElevate
)

$ErrorActionPreference = "Stop"

$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [System.Security.Principal.WindowsPrincipal]::new($identity)
$isAdministrator = $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdministrator -and -not $NoElevate) {
    $arguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$PSCommandPath`"",
        "-NoElevate"
    )
    $process = Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList $arguments `
        -WindowStyle Hidden -Wait -PassThru
    exit $process.ExitCode
}

$taskPath = "\SuperNovaBets\"
$xmlDir = Join-Path $PSScriptRoot "tasks"

$mlbTasks = @(
    "MLB-Morning",
    "MLB-PreGame-Day",
    "MLB-PreGame-Evening",
    "MLB-Prop-Targeted-Close",
    "MLB-Close",
    "MLB-Training"
)

$nflTasks = @(
    @{ Name = "NFL-Daily"; File = "NFL-Daily.xml" },
    @{ Name = "NFL-PrimeTime-Refresh"; File = "NFL-PrimeTime-Refresh.xml" },
    @{ Name = "NFL-Close"; File = "NFL-Close.xml" },
    @{ Name = "NFL-Training"; File = "NFL-Training.xml" },
    @{ Name = "NFL-Sharp-Watch"; File = "NFL-Sharp-Watch.xml" }
)

$failures = [System.Collections.Generic.List[string]]::new()

foreach ($name in $mlbTasks) {
    $task = Get-ScheduledTask -TaskPath $taskPath -TaskName $name -ErrorAction SilentlyContinue
    if (-not $task) {
        continue
    }
    try {
        if ($task.State -eq "Running") {
            Stop-ScheduledTask -TaskPath $taskPath -TaskName $name -ErrorAction Stop
        }
        Disable-ScheduledTask -TaskPath $taskPath -TaskName $name -ErrorAction Stop | Out-Null
        Write-Host "Disabled $taskPath$name"
    } catch {
        $failures.Add("Could not disable $taskPath$name`: $($_.Exception.Message)")
    }
}

foreach ($task in $nflTasks) {
    $xmlPath = Join-Path $xmlDir $task.File
    try {
        $xml = [System.IO.File]::ReadAllText($xmlPath, [System.Text.Encoding]::UTF8)
        $xml = $xml -replace '^\s*<\?xml[^?]*\?>\s*', ''
        Register-ScheduledTask -TaskPath $taskPath -TaskName $task.Name -Xml $xml -Force -ErrorAction Stop | Out-Null
        Enable-ScheduledTask -TaskPath $taskPath -TaskName $task.Name -ErrorAction Stop | Out-Null
        Write-Host "Registered/enabled $taskPath$($task.Name)"
    } catch {
        $failures.Add("Could not register/enable $taskPath$($task.Name): $($_.Exception.Message)")
    }
}

Write-Host ""
Get-ScheduledTask -TaskPath $taskPath |
    Where-Object { $_.TaskName -match '^(MLB|NFL)-' } |
    Sort-Object TaskName |
    Select-Object TaskName, State |
    Format-Table -AutoSize

if ($failures.Count -gt 0) {
    Write-Host ""
    Write-Warning "Completed with $($failures.Count) issue(s):"
    foreach ($failure in $failures) {
        Write-Warning $failure
    }
    exit 1
}

Write-Host "Done. MLB is disabled and NFL is enabled."
