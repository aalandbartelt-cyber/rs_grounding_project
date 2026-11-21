param(
    [string]$DataPath = 'D:\计算机视觉\rs_grounding_train_v1',
    [string]$OutputPath = 'D:\计算机视觉\Z_eval_outputs\v2'
)
$ErrorActionPreference = 'Stop'
$oldBytecodeSetting = $env:PYTHONDONTWRITEBYTECODE
$env:PYTHONDONTWRITEBYTECODE = '1'
function Invoke-CheckedPython {
    param([string[]]$PythonArgs)
    & python @PythonArgs
    if ($LASTEXITCODE -ne 0) { throw "Python check failed: $PythonArgs" }
}
try {
    $sourceRoot = [System.IO.Path]::GetFullPath($DataPath)
    $outputRoot = [System.IO.Path]::GetFullPath($OutputPath)
    $originalEval = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\rs_grounding_eval'))
    foreach ($protectedRoot in @($sourceRoot, $originalEval, [System.IO.Path]::GetFullPath($PSScriptRoot))) {
        if ($outputRoot.TrimEnd('\') -eq $protectedRoot.TrimEnd('\') -or $outputRoot.StartsWith($protectedRoot.TrimEnd('\') + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
            throw 'Choose an output directory outside the source dataset, original evaluation and code folder.'
        }
    }
    Invoke-CheckedPython -PythonArgs @('-m', 'unittest', 'discover', '-s', $PSScriptRoot, '-p', 'test_*.py', '-v')
    Invoke-CheckedPython -PythonArgs @((Join-Path $PSScriptRoot 'evalkit.py'), 'build', '--data', $sourceRoot, '--out', $outputRoot)
    Invoke-CheckedPython -PythonArgs @('-c', 'import json,sys; from pathlib import Path; a,b=[json.loads(Path(p).read_text(encoding="utf-8-sig")) for p in sys.argv[1:]]; assert a == b, "Built manifest differs from the committed v2 frozen manifest."', (Join-Path $PSScriptRoot 'splits\manifest.json'), (Join-Path $outputRoot 'splits\manifest.json'))
    Invoke-CheckedPython -PythonArgs @((Join-Path $PSScriptRoot 'evalkit.py'), 'baselines', '--root', $outputRoot)
    foreach ($protocol in @('dev4_81', 'official_val51_81')) {
        $inputDir = Join-Path $outputRoot "protocols\$protocol\input_path"
        $outputDir = Join-Path $outputRoot "runs\blind_random\$protocol"
        Invoke-CheckedPython -PythonArgs @((Join-Path $PSScriptRoot 'blind_baseline.py'), '--input_path', $inputDir, '--output_path', $outputDir, '--mode', 'random')
        Invoke-CheckedPython -PythonArgs @((Join-Path $PSScriptRoot 'evalkit.py'), 'score', '--gt', (Join-Path $outputRoot "private\${protocol}_gt.json"), '--pred', (Join-Path $outputDir 'pred.json'), '--out', (Join-Path $outputDir 'metrics.json'), '--require-complete')
    }
    Invoke-CheckedPython -PythonArgs @((Join-Path $PSScriptRoot 'evalkit.py'), 'verify-source', '--data', $sourceRoot, '--root', $outputRoot)
    Write-Output 'Z_ALL_CHECKS_PASSED'
} finally { $env:PYTHONDONTWRITEBYTECODE = $oldBytecodeSetting }
