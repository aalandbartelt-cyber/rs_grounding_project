param(
    [string]$DataPath='D:\计算机视觉\rs_grounding_train_v1',
    [string]$EvalPath='D:\计算机视觉\Z_eval_outputs\v2',
    [string]$OutputPath=('D:\计算机视觉\Z_model_lab\cpu_run_'+(Get-Date -Format 'yyyyMMdd_HHmmss')),
    [switch]$ExportTrainingAugmentations
)
$ErrorActionPreference='Stop'
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$outputRoot=[System.IO.Path]::GetFullPath($OutputPath)
foreach ($protected in @($DataPath,$EvalPath,$repoRoot)) {
    $resolved=[System.IO.Path]::GetFullPath($protected).TrimEnd('\')
    if ($outputRoot -eq $resolved -or $outputRoot.StartsWith($resolved+'\',[System.StringComparison]::OrdinalIgnoreCase)) { throw 'Output must be outside source, frozen evaluation and repository' }
}
if (Test-Path -LiteralPath $outputRoot) { throw 'Choose a new output directory to preserve prior results' }
New-Item -ItemType Directory -Path $outputRoot | Out-Null
function Invoke-StagePython {
    param([string[]]$ArgsList)
    & python @ArgsList
    if ($LASTEXITCODE -ne 0) { throw ('Stage failed: '+($ArgsList -join ' ')) }
}
Invoke-StagePython -ArgsList @('-m','unittest','discover','-s',(Join-Path $repoRoot 'z_evaluation'),'-p','test_*.py')
Invoke-StagePython -ArgsList @('-m','unittest','discover','-s',(Join-Path $repoRoot 'z_pipeline'),'-p','test_*.py')
Invoke-StagePython -ArgsList @('-m','unittest','discover','-s',$PSScriptRoot,'-p','test_*.py')
Invoke-StagePython -ArgsList @((Join-Path $repoRoot 'z_evaluation\evalkit.py'),'verify-source','--data',$DataPath,'--root',$EvalPath)
Invoke-StagePython -ArgsList @((Join-Path $PSScriptRoot 'audit_data.py'),'--data',$DataPath,'--out',(Join-Path $outputRoot 'audit'))
Invoke-StagePython -ArgsList @((Join-Path $PSScriptRoot 'preview.py'),'--data',$DataPath,'--out',(Join-Path $outputRoot 'previews'),'--per-class','1')
$inputDir=Join-Path $EvalPath 'protocols\dev4_81\input_path'
$fixture=Join-Path $outputRoot 'contract_fixture'
Invoke-StagePython -ArgsList @((Join-Path $PSScriptRoot 'contract.py'),'--input-path',$inputDir,'--output-path',$fixture,'--adapter','contract_fixture:create')
Invoke-StagePython -ArgsList @((Join-Path $PSScriptRoot 'diagnose.py'),'--input-path',$inputDir,'--gt',(Join-Path $EvalPath 'private\dev4_81_gt.json'),'--pred',(Join-Path $fixture 'pred.json'),'--candidates',(Join-Path $fixture 'candidates.json'),'--out',(Join-Path $outputRoot 'diagnostics_fixture'))
$diagnostics=Join-Path $outputRoot 'diagnostics_fixture\diagnostics.json'
Invoke-StagePython -ArgsList @((Join-Path $PSScriptRoot 'compare_runs.py'),'--a',$diagnostics,'--b',$diagnostics,'--out',(Join-Path $outputRoot 'comparison_smoke'))
Invoke-StagePython -ArgsList @((Join-Path $PSScriptRoot 'robust_inputs.py'),'--input-path',$inputDir,'--out',(Join-Path $outputRoot 'robustness_pilot'))
if ($ExportTrainingAugmentations) {
    $dataset=Join-Path $outputRoot 'augmented_detection_v1'
    Invoke-StagePython -ArgsList @((Join-Path $PSScriptRoot 'export_augmented.py'),'--data',$DataPath,'--out',$dataset)
    Invoke-StagePython -ArgsList @((Join-Path $PSScriptRoot 'validate_augmented.py'),'--data',$DataPath,'--dataset',$dataset,'--out',(Join-Path $outputRoot 'augmented_validation.json'))
}
Write-Output 'CPU_PREPARATION_CHECKS_PASSED_NO_MODEL_TRAINING'
