$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

python -m pytest -q
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

python scripts/04_audit.py
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
