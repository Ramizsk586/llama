$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ScriptPath = Join-Path $PSScriptRoot "llama_setup.py"
$OutputDir = Join-Path $RepoRoot "dist"
$IconPath = Join-Path $RepoRoot "assets\llama_bridge.ico"

python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -eq 0) {
  Write-Host "PyInstaller already installed; skipping pip install."
} else {
  python -m pip install --upgrade pyinstaller
}

python -m PyInstaller `
  --noconfirm `
  --clean `
  --onefile `
  --console `
  --name "llama setup" `
  --icon $IconPath `
  --distpath $OutputDir `
  --workpath (Join-Path $RepoRoot "build\setup") `
  --specpath (Join-Path $RepoRoot "build\setup") `
  $ScriptPath

$BuiltExe = Join-Path $OutputDir "llama setup.exe"
$FinalExe = Join-Path $OutputDir "llama setup.exe"
if (-not (Test-Path $BuiltExe)) {
  throw "Expected setup exe was not created: $BuiltExe"
}

Write-Host "Created $FinalExe"
