param(
  [string]$BuildDir = "build-win",
  [int]$OmpNumThreads = 4,
  [switch]$Wipe
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildPath = Join-Path $RepoRoot $BuildDir
$OpenBlasLibPath =
    Join-Path $RepoRoot "subprojects\nntrainer\nntrainer-windows-resource\x64\OpenBLAS"

function Resolve-RequiredCommand($Name) {
  $cmd = Get-Command $Name -ErrorAction SilentlyContinue
  if ($cmd) {
    return $cmd.Source
  }

  throw "Cannot find $Name. Install it or add it to PATH."
}

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
  throw "cl.exe is not in PATH. Run this script from a VS 2022 x64 Developer PowerShell/Command Prompt."
}

$git = Resolve-RequiredCommand "git.exe"

if ($Wipe -and (Test-Path $BuildPath)) {
  Remove-Item -LiteralPath $BuildPath -Recurse -Force
}

& $git -C $RepoRoot submodule update --init --recursive

if (Test-Path $OpenBlasLibPath) {
  $env:LIB = "$OpenBlasLibPath;$env:LIB"
}

$mesonSetupArgs = @(
  "setup",
  $BuildPath,
  "-Dplatform=windows",
  "-Denable-fp16=false",
  "-Denable-openmp=true",
  "-Dthread-backend=omp",
  "-Domp-num-threads=$OmpNumThreads",
  "-Dcpp_std=c++20",
  "-Dnntrainer:cpp_std=c++20",
  "-Dnntrainer:platform=windows"
)

if (Test-Path (Join-Path $BuildPath "meson-info")) {
  $mesonSetupArgs += "--reconfigure"
}

& py -m mesonbuild.mesonmain @mesonSetupArgs

$NntrainerSubprojectBuild = Join-Path $BuildPath "subprojects\nntrainer\nntrainer"
$NntrainerTopLevelCompat = Join-Path $BuildPath "nntrainer"

if ((Test-Path $NntrainerSubprojectBuild) -and
    (-not (Test-Path $NntrainerTopLevelCompat))) {
  New-Item -ItemType Junction `
    -Path $NntrainerTopLevelCompat `
    -Target $NntrainerSubprojectBuild | Out-Null
}

& py -m ninja -C $BuildPath
