param()

$ErrorActionPreference = "Stop"
$ProviderDir = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $ProviderDir "..\..\..")).Path
$ModelRoot = Join-Path $RepoRoot ".models\sam2_video"
$ModelRepo = Join-Path $ModelRoot "repo"
$CheckpointDir = Join-Path $ModelRoot "checkpoints"
$Checkpoint = Join-Path $CheckpointDir "sam2.1_hiera_small.pt"
$Venv = Join-Path $RepoRoot ".venvs\sam2_video"
$Python = Join-Path $Venv "Scripts\python.exe"
$OfficialCommit = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
$CheckpointUrl = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
$CheckpointSha256 = "6D1AA6F30DE5C92224F8172114DE081D104BBD23DD9DC5C58996F0CAD5DC4D38"

New-Item -ItemType Directory -Force -Path $ModelRoot, $CheckpointDir | Out-Null

if (-not (Test-Path (Join-Path $ModelRepo ".git"))) {
    & git clone --filter=blob:none https://github.com/facebookresearch/sam2.git $ModelRepo
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}

$ActualCommit = (& git -C $ModelRepo rev-parse HEAD).Trim()
if ($ActualCommit -ne $OfficialCommit) {
    & git -C $ModelRepo fetch origin $OfficialCommit --depth 1
    if ($LASTEXITCODE -ne 0) { throw "git fetch of pinned commit failed" }
    & git -C $ModelRepo checkout --detach $OfficialCommit
    if ($LASTEXITCODE -ne 0) { throw "git checkout of pinned commit failed" }
}

if (-not (Test-Path $Python)) {
    & py -3.11 -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}

& $Python -m pip install --no-cache-dir --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed" }
& $Python -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }
& $Python -m pip install --no-cache-dir hydra-core==1.3.2 iopath==0.1.10 tqdm==4.67.1 opencv-python-headless==4.13.0.92
if ($LASTEXITCODE -ne 0) { throw "SAM 2 runtime dependency install failed" }

if (-not (Test-Path $Checkpoint)) {
    & curl.exe -L --fail --retry 3 --output $Checkpoint $CheckpointUrl
    if ($LASTEXITCODE -ne 0) { throw "checkpoint download failed" }
}
$ActualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Checkpoint).Hash
if ($ActualHash -ne $CheckpointSha256) {
    throw "checkpoint SHA-256 mismatch: $ActualHash"
}

& $Python -c "import torch, torchvision, cv2; assert torch.cuda.is_available(); print(torch.__version__, torchvision.__version__, cv2.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
if ($LASTEXITCODE -ne 0) { throw "CUDA smoke test failed" }
