[CmdletBinding()]
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProviderDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ProviderDir "..\..\..")).Path
$ModelBase = Join-Path $RepoRoot ".models\video_matting"
$ModelRepo = Join-Path $ModelBase "MatAnyone"
$CheckpointDir = Join-Path $ModelBase "checkpoints"
$Checkpoint = Join-Path $CheckpointDir "matanyone.pth"
$Venv = Join-Path $RepoRoot ".venvs\video_matting"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$TempDir = Join-Path $ModelBase "tmp"

$Commit = "e5ddc534c1fff9bb9e54cf476095d29071b7cb4f"
$CheckpointUrl = "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth"
$CheckpointSha256 = "dd26b991d020ed5eb4be50996f97354c45cfdfc0f59958e8983ac6a198f4809d"

New-Item -ItemType Directory -Force -Path $ModelBase, $CheckpointDir, $TempDir | Out-Null
$env:TEMP = $TempDir
$env:TMP = $TempDir
$env:PIP_CACHE_DIR = Join-Path $ModelBase "pip-cache"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:TORCH_HOME = Join-Path $ModelBase "torch-cache"
$env:HF_HOME = Join-Path $ModelBase "hf-cache"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    & $Python -m venv $Venv
}

& $VenvPython -m pip install --upgrade "pip==25.1.1" "setuptools==80.9.0" "wheel==0.45.1"
& $VenvPython -m pip install --index-url https://download.pytorch.org/whl/cu124 `
    "torch==2.5.1+cu124" "torchvision==0.20.1+cu124"
& $VenvPython -m pip install `
    "numpy==1.26.4" `
    "Pillow==10.4.0" `
    "opencv-python-headless==4.10.0.84" `
    "omegaconf==2.3.0" `
    "hydra-core==1.3.2" `
    "huggingface-hub==0.27.1" `
    "safetensors==0.5.2" `
    "tqdm==4.67.1" `
    "imageio==2.25.0"

if (-not (Test-Path -LiteralPath (Join-Path $ModelRepo ".git"))) {
    git clone --filter=blob:none https://github.com/pq-yang/MatAnyone.git $ModelRepo
}
$ActualCommit = (git -C $ModelRepo rev-parse HEAD).Trim()
if ($ActualCommit -ne $Commit) {
    $ModelChanges = git -C $ModelRepo status --porcelain
    if ($ModelChanges) {
        throw "MatAnyone checkout has local changes; refusing to switch commits."
    }
    git -C $ModelRepo fetch origin
    git -C $ModelRepo checkout --detach $Commit
}

$NeedsCheckpoint = $true
if (Test-Path -LiteralPath $Checkpoint) {
    $CurrentHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Checkpoint).Hash.ToLowerInvariant()
    $NeedsCheckpoint = $CurrentHash -ne $CheckpointSha256
}
if ($NeedsCheckpoint) {
    $Part = "$Checkpoint.part"
    if (Test-Path -LiteralPath $Part) {
        Remove-Item -LiteralPath $Part -Force
    }
    & curl.exe --fail --location --retry 3 --output $Part $CheckpointUrl
    $DownloadedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Part).Hash.ToLowerInvariant()
    if ($DownloadedHash -ne $CheckpointSha256) {
        throw "Checkpoint SHA-256 mismatch: $DownloadedHash"
    }
    Move-Item -LiteralPath $Part -Destination $Checkpoint -Force
}

& $VenvPython -m pip check
Write-Host "MatAnyone runtime is ready."
Write-Host "Source: $ModelRepo @ $Commit"
Write-Host "Checkpoint: $Checkpoint ($CheckpointSha256)"
