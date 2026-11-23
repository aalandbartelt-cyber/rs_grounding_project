param([string]$LabRoot = 'D:\计算机视觉\Z_model_lab')
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HF_HOME = Join-Path $LabRoot 'cache\huggingface'
$env:PIP_CACHE_DIR = Join-Path $LabRoot 'cache\pip'
$env:HF_HUB_DISABLE_XET = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:TEMP = Join-Path $LabRoot 'temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $LabRoot,$env:TEMP,$env:PIP_CACHE_DIR,$env:HF_HOME | Out-Null
$venvRoot = Join-Path $LabRoot 'venv'
$pythonExe = Join-Path $venvRoot 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    & python -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
}
& $pythonExe -m pip install --disable-pip-version-check --progress-bar off 'torch==2.6.0' 'torchvision==0.21.0' --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
& $pythonExe -m pip install --disable-pip-version-check --progress-bar off -r (Join-Path $PSScriptRoot 'requirements-baseline.txt')
if ($LASTEXITCODE -ne 0) { throw 'Baseline dependencies installation failed' }
& $pythonExe -c 'import torch; print(torch.__version__); assert torch.cuda.is_available(), "CUDA unavailable"; x=torch.randn(256,256,device="cuda"); print(torch.cuda.get_device_name(0), (x@x).mean().item())'
if ($LASTEXITCODE -ne 0) { throw 'CUDA smoke test failed' }
& $pythonExe -m pip freeze | Set-Content -Encoding UTF8 (Join-Path $LabRoot 'environment-lock.txt')
