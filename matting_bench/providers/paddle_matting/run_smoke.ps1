param(
    [string]$SourceDir,
    [ValidateSet("cuda", "gpu", "cpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$ProviderDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ProviderDir "..\..\..")
$Python = Join-Path $RepoRoot ".venvs\paddle_matting\Scripts\python.exe"

if (-not $SourceDir) {
    $SourceDir = Join-Path $RepoRoot "poc_output\pet_20260710_121221_5ce7716e_real_after"
}

$FramesDir = Join-Path $ProviderDir "evidence\frames"
$OutputDir = Join-Path $ProviderDir "evidence\rgba_$Device"

$ErrorActionPreference = "Continue"
& $Python (Join-Path $ProviderDir "extract_smoke_frames.py") `
    --source-dir $SourceDir `
    --output-dir $FramesDir
if ($LASTEXITCODE -ne 0) {
    throw "frame extraction failed with exit code $LASTEXITCODE"
}

& $Python (Join-Path $ProviderDir "infer.py") `
    --input-dir $FramesDir `
    --output-dir $OutputDir `
    --device $Device `
    --warmup-runs 1
if ($LASTEXITCODE -ne 0) {
    throw "Paddle matting inference failed with exit code $LASTEXITCODE"
}
$ErrorActionPreference = "Stop"
