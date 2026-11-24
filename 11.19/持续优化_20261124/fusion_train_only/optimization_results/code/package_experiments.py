"""Archive audited experiment source and records without image data or model blobs."""
import hashlib
import json
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

BASE = Path(__file__).resolve().parents[2]
PROJECT = BASE.parent
DEST = PROJECT/'交付/11.19_持续优化_20261124'


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    if DEST.exists() and '--rebuild' not in sys.argv:
        raise FileExistsError(f'Archive directory already exists: {DEST}')
    if DEST.exists():
        existing = json.loads((DEST/'artifact_manifest.json').read_text(encoding='utf-8'))
        assert existing['artifact_type'] == 'experiment source and records; not a submission image or official weight export'
    verification = json.loads((BASE/'optimization_results/verification.json').read_text(encoding='utf-8'))
    assert verification['score_arithmetic_for_all_rounds_verified']
    assert verification['official_validation_or_test_used'] is False
    best = json.loads((BASE/'optimization_results/current_best.json').read_text(encoding='utf-8'))
    packages = {}
    for name in ('torch', 'torchvision', 'transformers', 'open_clip_torch', 'scikit-learn',
                 'joblib', 'numpy', 'Pillow', 'huggingface-hub', 'accelerate',
                 'bitsandbytes', 'sentencepiece', 'safetensors'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    environment = {'python': '3.12', 'device': 'NVIDIA GeForce RTX 4060 Laptop GPU 8 GiB',
                   'A100_rerun_performed': False, 'packages': packages}
    (BASE/'environment_packages.json').write_text(json.dumps(environment, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    paths = [BASE/'README.md', BASE/'复现步骤_20261124.md', BASE/'environment_packages.json', BASE/'submission_limits.json',
             BASE/'data/manifest.json', BASE/'sources/repositories.json']
    for folder in [BASE/'code']+[BASE/f'training_v{number}/code' for number in range(1, 19)]+[BASE/'optimization_results/code']:
        if folder.is_dir():
            paths.extend(path for path in folder.iterdir() if path.is_file() and path.suffix in ('.py', '.ps1'))
    for folder in [BASE/'results', BASE/'optimization_results']+[BASE/f'training_v{number}/results' for number in range(1, 19)]:
        if folder.is_dir():
            paths.extend(path for path in folder.iterdir()
                         if path.is_file() and path.suffix in ('.json', '.csv', '.md', '.log')
                         and not path.name.startswith('all49_'))
    for number in range(1, 19):
        folder = BASE/f'training_v{number}'
        paths.extend(path for path in folder.glob('*.log') if path.is_file())
    manifest = []
    for source in sorted(set(paths)):
        relative = source.relative_to(BASE)
        target = DEST/'fusion_train_only'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha256(source) == sha256(target)
        manifest.append({'path': target.relative_to(DEST).as_posix(),
                         'bytes': target.stat().st_size, 'sha256': sha256(target)})

    summary = {
        'artifact_type': 'experiment source and records; not a submission image or official weight export',
        'development_score_out_of_1': best['score_out_of_1'], 'target_score': .6,
        'target_met': best['target_met'], 'official_validation_or_test_used': False,
        'images_included': False, 'model_weight_blobs_included': False,
        'full_feature_caches_included': False, 'files': manifest}
    (DEST/'artifact_manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (DEST/'README.md').write_text(
        '# 11.19 后续融合实验：2026-11-24\n\n'
        '当前保留内部开发总分 **0.5259/1**，尚未达到 0.6。仅使用 49 张公开原始训练图，'
        '33 张拟合、16 张内部开发；没有读取官方验证或测试图像，也没有用旧 47 张留出图训练或评分。\n\n'
        '- [逐轮总分与核验](fusion_train_only/optimization_results/README.md)\n'
        '- [分步运行说明和实际结果](fusion_train_only/复现步骤_20261124.md)\n'
        '- [当前结果索引](fusion_train_only/optimization_results/current_best.json)\n'
        '- [数据访问和哈希核验](fusion_train_only/optimization_results/verification.json)\n\n'
        '本目录保存代码、日志与实验记录。图像、模型大权重和完整中间特征缓存保留在 E 盘原项目，'
        '不包含在本归档中。它不是自包含训练包、Docker 镜像或通过 0.6 门槛的权重导出包。'
        '本轮设备为 RTX 4060 Laptop；A100 本轮未实测，原 A100 环境记录继续单独归档。\n', encoding='utf-8')
    zip_path = DEST.parent/(DEST.name+'.zip')
    with ZipFile(zip_path, 'w', compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for source in sorted(DEST.rglob('*')):
            if source.is_file():
                archive.write(source, source.relative_to(DEST).as_posix())
    with ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        assert not any(Path(name).suffix.lower() in ('.png', '.jpg', '.jpeg', '.safetensors', '.pt', '.bin') for name in archive.namelist())
    checksum = zip_path.with_suffix('.zip.sha256')
    checksum.write_text(sha256(zip_path)+'  '+zip_path.name+'\n', encoding='utf-8')
    print(json.dumps({'directory': str(DEST), 'zip': str(zip_path), 'files': len(manifest)+2,
                      'bytes': zip_path.stat().st_size, 'sha256': sha256(zip_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
