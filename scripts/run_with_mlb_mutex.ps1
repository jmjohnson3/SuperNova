param(
    [string]$LockName = "SuperNovaBets_MLB_Operational",
    [string]$CommandText
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($CommandText)) {
    Write-Error "No command supplied to run_with_mlb_mutex.ps1"
    exit 64
}

$createdNew = $false
$mutex = [System.Threading.Mutex]::new($true, $LockName, [ref]$createdNew)

if (-not $createdNew) {
    Write-Host "Another MLB operational task is already running under mutex '$LockName'. Skipping this scheduled run."
    exit 0
}

try {
    Write-Host "Acquired MLB task mutex '$LockName'."
    cmd.exe /d /s /c $CommandText
    $code = if ($LASTEXITCODE -ne $null) { [int]$LASTEXITCODE } else { 0 }
    exit $code
}
finally {
    try {
        $mutex.ReleaseMutex() | Out-Null
    } finally {
        $mutex.Dispose()
    }
}
