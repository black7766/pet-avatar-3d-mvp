[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [string]$OutputDir,
    [int]$Frames = 24,
    [int]$FrameOffset = 0,
    [int]$MaxSize = 640,
    [int]$Warmup = 10,
    [ValidateSet("alpha", "mask")]
    [string]$InitKind = "mask",
    [ValidateSet("green-clean", "source")]
    [string]$RgbaRgb = "green-clean",
    [switch]$Overwrite,
    [switch]$NoAmp
)

$ErrorActionPreference = "Stop"
$ProviderDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ProviderDir "..\..\..")).Path
$ModelBase = Join-Path $RepoRoot ".models\video_matting"
$VenvPython = Join-Path $RepoRoot ".venvs\video_matting\Scripts\python.exe"
$Cli = Join-Path $ProviderDir "matanyone_cli.py"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Missing video-matting environment. Run $ProviderDir\setup.ps1 first."
}

if (-not $OutputDir) {
    $OutputDir = Join-Path $ProviderDir "runs\matanyone"
}

$env:TEMP = Join-Path $ModelBase "tmp"
$env:TMP = $env:TEMP
$env:TORCH_HOME = Join-Path $ModelBase "torch-cache"
$env:HF_HOME = Join-Path $ModelBase "hf-cache"
$env:PYTHONPYCACHEPREFIX = Join-Path $RepoRoot ".venvs\video_matting\pycache"

$Arguments = @(
    $Cli,
    "--input", $InputPath,
    "--output-dir", $OutputDir,
    "--frames", $Frames,
    "--frame-offset", $FrameOffset,
    "--max-size", $MaxSize,
    "--warmup", $Warmup,
    "--init-kind", $InitKind,
    "--rgba-rgb", $RgbaRgb
)
if ($Overwrite) {
    $Arguments += "--overwrite"
}
if ($NoAmp) {
    $Arguments += "--no-amp"
}

& $VenvPython @Arguments
exit $LASTEXITCODE
