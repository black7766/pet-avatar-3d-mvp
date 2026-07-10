param(
    [string]$InputDir,
    [string]$OutputDir,
    [int]$Frames = 24,
    [int]$FrameOffset = 0,
    [int]$MaskThreshold = 128,
    [ValidateSet("fp16", "fp32")]
    [string]$Precision = "fp16",
    [switch]$OffloadStateToCpu,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProviderDir = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $ProviderDir "..\..\..")).Path
$Python = Join-Path $RepoRoot ".venvs\sam2_video\Scripts\python.exe"
if (-not $InputDir) {
    $InputDir = Join-Path $RepoRoot "matting_bench\data\pet_20260710_121221_5ce7716e\full\fast_walk"
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $ProviderDir "runs\fast_walk_24_sam2_1_small"
}

$Arguments = @(
    (Join-Path $ProviderDir "infer.py"),
    "--input-dir", $InputDir,
    "--output-dir", $OutputDir,
    "--frames", $Frames,
    "--frame-offset", $FrameOffset,
    "--mask-threshold", $MaskThreshold,
    "--precision", $Precision
)
if ($OffloadStateToCpu) { $Arguments += "--offload-state-to-cpu" }
if ($Overwrite) { $Arguments += "--overwrite" }

& $Python @Arguments
exit $LASTEXITCODE
