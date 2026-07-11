param(
    [string]$InputDir,
    [string]$OutputDir,
    [int]$Frames = 24,
    [int]$FrameOffset = 0,
    [int]$MaskThreshold = 128,
    [double]$LogitThreshold = 0.0,
    [ValidateSet("fp16", "fp32")]
    [string]$Precision = "fp16",
    [switch]$VideoOnGpu,
    [switch]$OffloadStateToCpu,
    [switch]$AsyncLoadingFrames,
    [switch]$NoPostprocessing,
    [switch]$VosOptimized,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProviderDir = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $ProviderDir "..\..\..")).Path
$Python = Join-Path $RepoRoot ".venvs\sam2_video\Scripts\python.exe"
if (-not $InputDir) {
    $InputDir = Join-Path $RepoRoot "matting_bench\data\pet_20260710_121221_5ce7716e\temporal_fast_walk_24_640"
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
    "--logit-threshold", $LogitThreshold,
    "--precision", $Precision
)
if ($VideoOnGpu) { $Arguments += "--no-offload-video-to-cpu" }
if ($OffloadStateToCpu) { $Arguments += "--offload-state-to-cpu" }
if ($AsyncLoadingFrames) { $Arguments += "--async-loading-frames" }
if ($NoPostprocessing) { $Arguments += "--no-apply-postprocessing" }
if ($VosOptimized) { $Arguments += "--vos-optimized" }
if ($Overwrite) { $Arguments += "--overwrite" }

& $Python @Arguments
exit $LASTEXITCODE
