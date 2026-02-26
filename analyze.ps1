# ============================================================
#  Spanish Podcast Difficulty Analyzer - Setup & Run Script
#  Creates a virtualenv, installs dependencies, and runs
#  the analyzer with any arguments you pass in.
# ============================================================

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvDir = Join-Path $ScriptDir '.venv'
$Requirements = Join-Path $ScriptDir 'requirements.txt'

# ----------------------------------------------------------
#  1. Find Python
# ----------------------------------------------------------
$Python = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $Python = 'python'
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $Python = 'python3'
} else {
    Write-Error '[ERROR] Python not found. Please install Python 3.10+ and add it to PATH.'
    exit 1
}

Write-Host "[SETUP] Using Python: $(& $Python --version 2>&1)"

# ----------------------------------------------------------
#  2. Create virtualenv if it doesn't exist
# ----------------------------------------------------------
$ActivateScript = Join-Path $VenvDir 'Scripts\Activate.ps1'
if (-not (Test-Path $ActivateScript)) {
    Write-Host '[SETUP] Creating virtual environment in' $VenvDir '...'
    & $Python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error '[ERROR] Failed to create virtual environment.'
        exit 1
    }
}

# ----------------------------------------------------------
#  3. Activate virtualenv
# ----------------------------------------------------------
& $ActivateScript

# ----------------------------------------------------------
#  4. Install / upgrade dependencies
# ----------------------------------------------------------
# We use a stamp file so we only pip-install when requirements.txt changes.
$Stamp = Join-Path $VenvDir '.deps_installed'

$NeedsInstall = $false
if (-not (Test-Path $Stamp)) {
    $NeedsInstall = $true
} else {
    $ReqDate = (Get-Item $Requirements).LastWriteTime
    $StampDate = (Get-Item $Stamp).LastWriteTime
    if ($ReqDate -gt $StampDate) {
        $NeedsInstall = $true
    }
}

if ($NeedsInstall) {
    Write-Host '[SETUP] Installing Python dependencies ...'
    python -m pip install --upgrade pip 2>&1 | Out-Null
    python -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) {
        Write-Error '[ERROR] pip install failed.'
        exit 1
    }

    Write-Host '[SETUP] Downloading spaCy Spanish model ...'
    python -m spacy download es_core_news_lg
    if ($LASTEXITCODE -ne 0) {
        Write-Warning '[WARNING] spaCy model download failed. Structural analysis may not work.'
    }

    # Write stamp file
    'installed' | Out-File -FilePath $Stamp -Encoding utf8
    Write-Host '[SETUP] Dependencies installed successfully.'
} else {
    Write-Host '[SETUP] Dependencies up to date.'
}

# ----------------------------------------------------------
#  5. Run the analyzer (pass through all script arguments)
# ----------------------------------------------------------
if ($args.Count -eq 0) {
    Write-Host ''
    Write-Host 'Usage:  .\analyze.ps1 [OPTIONS] FEED_URL [FEED_URL ...]'
    Write-Host ''
    Write-Host 'Run ".\analyze.ps1 --help" for full options.'
    Write-Host ''
    exit 0
}

Write-Host ''
Write-Host '[RUN] Starting analyzer ...'
Write-Host ''
python (Join-Path $ScriptDir 'main.py') @args
