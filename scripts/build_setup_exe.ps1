# Changelog:
# - Added colored build output, Python 3.11+ preflight checks, and try/catch failure handling.
# - Added graceful icon fallback when assets\llama_bridge.ico is unavailable.
# - Added PyInstaller output verification, file-size reporting, and a final build summary.

$ErrorActionPreference = "Stop"

function Write-Step {
  param([string]$Message)
  Write-Host "[STEP] $Message" -ForegroundColor Cyan
}

function Write-Ok {
  param([string]$Message)
  Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warn {
  param([string]$Message)
  Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Fail {
  param([string]$Message)
  Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Invoke-Python {
  param(
    [string[]]$Arguments,
    [switch]$Capture
  )
  if ($script:UseUvPython) {
    $UvArgs = @(
      "run",
      "--no-project",
      "--managed-python",
      "--python", "3.12",
      "--with", "pyinstaller",
      "python"
    ) + $Arguments
    if ($Capture) {
      return & uv @UvArgs
    }
    & uv @UvArgs
    return
  }
  if ($Capture) {
    return & $script:PythonExe @Arguments
  }
  & $script:PythonExe @Arguments
}

try {
  $RepoRoot = Split-Path -Parent $PSScriptRoot
  $ScriptPath = Join-Path $PSScriptRoot "llama_setup.py"
  $OutputDir = Join-Path $RepoRoot "dist"
  $IconPath = Join-Path $RepoRoot "assets\llama_bridge.ico"
  $env:UV_CACHE_DIR = Join-Path $RepoRoot ".uv-build-cache"
  $env:UV_PYTHON_INSTALL_DIR = Join-Path $RepoRoot ".uv-python"

  $script:UseUvPython = $false
  $script:PythonExe = $null
  $PythonCandidates = @(
    Get-Command python -All -ErrorAction SilentlyContinue |
      Select-Object -ExpandProperty Source -Unique |
      Where-Object { $_ -notlike "*\WindowsApps\*" -and $_ -notlike "*\hermes-agent\venv\*" }
  )
  foreach ($Candidate in $PythonCandidates) {
    try {
      $VersionProbe = & $Candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'); raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" 2>$null
    } catch {
      continue
    }
    if ($LASTEXITCODE -eq 0) {
      $script:PythonExe = $Candidate
      break
    }
  }
  if (-not $script:PythonExe) {
    Write-Warn "No healthy python.exe found on PATH; using uv managed Python."
    $script:UseUvPython = $true
  }

  Write-Step "Checking Python version"
  $PythonVersion = Invoke-Python -Capture -Arguments @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'); raise SystemExit(0 if sys.version_info >= (3, 11) else 1)")
  if ($LASTEXITCODE -ne 0) {
    throw "Python 3.11 or newer is required to build the setup executable."
  }
  Write-Ok "Found Python $PythonVersion"

  Write-Step "Checking PyInstaller"
  Invoke-Python -Arguments @("-c", "import PyInstaller") 2>$null
  if ($LASTEXITCODE -eq 0) {
    Write-Ok "PyInstaller already installed; skipping pip install."
  } else {
    if ($script:UseUvPython) {
      throw "PyInstaller should have been provided by uv but could not be imported."
    }
    Write-Step "Installing PyInstaller"
    Invoke-Python -Arguments @("-m", "pip", "install", "--upgrade", "pyinstaller")
    if ($LASTEXITCODE -ne 0) {
      throw "Failed to install PyInstaller."
    }
  }

  $PyInstallerArgs = @(
    "--noconfirm",
    "--clean",
    "--onefile",
    "--console",
    "--name", "llama setup"
  )

  if (Test-Path -LiteralPath $IconPath) {
    $PyInstallerArgs += @("--icon", $IconPath)
    Write-Ok "Using icon: $IconPath"
  } else {
    Write-Warn "Icon not found at $IconPath; building without --icon."
  }

  $PyInstallerArgs += @(
    "--distpath", $OutputDir,
    "--workpath", (Join-Path $RepoRoot "build\setup"),
    "--specpath", (Join-Path $RepoRoot "build\setup"),
    $ScriptPath
  )

  Write-Step "Building llama setup.exe"
  Invoke-Python -Arguments (@("-m", "PyInstaller") + $PyInstallerArgs)
  if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed."
  }

  $BuiltExe = Join-Path $OutputDir "llama setup.exe"
  if (-not (Test-Path -LiteralPath $BuiltExe)) {
    throw "Expected setup exe was not created: $BuiltExe"
  }

  $Item = Get-Item -LiteralPath $BuiltExe
  $SizeMb = [math]::Round($Item.Length / 1MB, 2)
  Write-Ok "Created $BuiltExe ($SizeMb MB)"

  Write-Host ""
  Write-Host "Build summary" -ForegroundColor Cyan
  Write-Host "-------------" -ForegroundColor Cyan
  Write-Host "Output: $BuiltExe" -ForegroundColor Green
  Write-Host "Size  : $SizeMb MB" -ForegroundColor Green
} catch {
  Write-Fail $_.Exception.Message
  exit 1
}
