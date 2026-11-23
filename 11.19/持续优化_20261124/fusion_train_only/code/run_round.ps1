$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$experimentPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
$experimentLogs = Join-Path $projectRoot 'results/logs'
New-Item -ItemType Directory -Path $experimentLogs -Force | Out-Null

function Run-Step {
    param([string]$Script, [string[]]$Arguments = @())
    $stepLog = Join-Path $experimentLogs ($Script.Replace('.py', '') + '_' + ($Arguments -join '_').Replace('-', '') + '.log')
    "$(Get-Date -Format o) Starting $Script $($Arguments -join ' ')" | Tee-Object -FilePath $stepLog -Append
    & $experimentPython (Join-Path $PSScriptRoot $Script) @Arguments *>&1 | Tee-Object -FilePath $stepLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment step failed: $Script"
    }
}

$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
Run-Step -Script 'translate.py'
Run-Step -Script 'encode_retrieval.py' -Arguments @('RemoteCLIP', '--text-only')
Run-Step -Script 'encode_retrieval.py' -Arguments @('GeoRSCLIP', '--text-only')
Run-Step -Script 'features.py'
Run-Step -Script 'ground_geoground.py'
Run-Step -Script 'box_parser.py'
Run-Step -Script 'encode_geo_boxes.py'
Run-Step -Script 'rerank_qwen.py'
Run-Step -Script 'score_round.py'
Run-Step -Script 'write_report.py'
Run-Step -Script 'verify_round.py'
