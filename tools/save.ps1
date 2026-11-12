param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Message
)

# One-command snapshot: stage everything, commit with a timestamp, show recent history.
# Usage (PowerShell 5.1 or 7):
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\save.ps1 "what you changed"

$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo

git add -A

$staged = @(git diff --cached --name-only)
if ($staged.Count -eq 0) {
    Write-Host "Nothing to commit - working tree is clean."
    exit 0
}

Write-Host ("Staged {0} file(s):" -f $staged.Count)
$staged | ForEach-Object { "  $_" }

git commit -m "$Message  [$(Get-Date -Format 'yyyy-MM-dd HH:mm')]"

Write-Host ""
Write-Host "Last 3 commits:"
git log --oneline -3
