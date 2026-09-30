param(
    [string]$LockName = "SuperNovaBets_MLB_Operational",
    [string]$CommandText,
    [int]$WaitSeconds = 3600
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($CommandText)) {
    Write-Error "No command supplied to run_with_mlb_mutex.ps1"
    exit 64
}

$mutex = [System.Threading.Mutex]::new($false, $LockName)
$acquired = $false

try {
    try {
        $acquired = $mutex.WaitOne([TimeSpan]::FromSeconds([Math]::Max(0, $WaitSeconds)))
    } catch [System.Threading.AbandonedMutexException] {
        $acquired = $true
        Write-Warning "Recovered abandoned MLB task mutex '$LockName'."
    }
    if (-not $acquired) {
        [Console]::Error.WriteLine("Timed out after ${WaitSeconds}s waiting for MLB task mutex '$LockName'.")
        exit 75
    }
    Write-Host "Acquired MLB task mutex '$LockName'."
    cmd.exe /d /s /c $CommandText
    $code = if ($LASTEXITCODE -ne $null) { [int]$LASTEXITCODE } else { 0 }
    exit $code
}
finally {
    try {
        if ($acquired) {
            $mutex.ReleaseMutex() | Out-Null
        }
    } finally {
        $mutex.Dispose()
    }
}
