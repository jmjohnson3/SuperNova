param()
$ErrorActionPreference = 'Stop'
$taskPath = '\SuperNovaBets\'
$name = 'NFL-PrimeTime-Refresh'
$path = Join-Path $PSScriptRoot 'tasks\NFL-PrimeTime-Refresh.xml'
$xml = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)
$xml = $xml -replace '^\s*<\?xml[^?]*\?>\s*', ''
# Update the existing refresh task, rather than install a duplicate publisher.
Register-ScheduledTask -TaskPath $taskPath -TaskName $name -Xml $xml -Force | Out-Null
Enable-ScheduledTask -TaskPath $taskPath -TaskName $name | Out-Null
$task = Get-ScheduledTask -TaskPath $taskPath -TaskName $name
$info = Get-ScheduledTaskInfo -InputObject $task
[pscustomobject]@{
    Task = $task.TaskName
    State = $task.State
    Interval = ($task.Triggers.Repetition.Interval -join ',')
    Arguments = ($task.Actions.Arguments -join ',')
    NextRun = $info.NextRunTime
} | Format-List
