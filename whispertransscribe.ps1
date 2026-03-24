[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:UV_CACHE_DIR = Join-Path $repoRoot ".uv-cache"

Push-Location $repoRoot
try {
    & uv run .\main.py @Arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
