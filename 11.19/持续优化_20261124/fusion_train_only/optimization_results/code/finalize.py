"""Verify and publish comparable development scores, preserving every experiment."""
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import joblib
import torch

BASE = Path(__file__).resolve().parents[2]
OUT = BASE/'optimization_results'
sys.path.insert(0, str(BASE/'training_v2/code'))
from inputs import Data, TrainArchive, digest, write_json
from score_round import measure
from train import chosen_boxes
from lora_visual import combined_scores
import retrieval


def read(path):
    return json.loads((BASE/path).read_text(encoding='utf-8'))


def lora_boxes(data):
    visual = torch.load(BASE/'training_v1/features/RemoteCLIP_lora_dev_regions.pt', weights_only=True)
    return chosen_boxes(data, data.dev_ids, combined_scores(data, data.dev_ids, visual), .95)


def reproduce(method, data):
    if method == 'v1_lora':
        from evaluate import predict
        previous = torch.load(BASE/'training_v1/weights/trained_fusion_final.pt', weights_only=True)
        result = predict(data, data.dev_ids, previous)
        result['grounding'] = lora_boxes(data)
        return result
    if method == 'v2_retained':
        bundle = joblib.load(BASE/'training_v2/weights/multiview_candidate_ranker.joblib')
        rankings = retrieval.predict(data, retrieval.features(data), data.dev_ids,
                                     bundle['retrieval_model'], bundle['config']['retrieval'])
        return {'grounding': lora_boxes(data), 'rankings': rankings}
    if method == 'v7_geometry':
        sys.path.insert(0, str(BASE/'training_v7/code'))
        import refine_boxes
        features = refine_boxes.fusion.ComplementaryFeatures()
        visual = torch.load(BASE/'training_v1/features/RemoteCLIP_lora_dev_regions.pt', weights_only=True)
        selected = {key: int(value.argmax()) for key, value in combined_scores(data, data.dev_ids, visual).items()}
        correction = joblib.load(BASE/'training_v7/weights/geometry_correction.joblib')
        boxes = refine_boxes.predict(features, data.dev_ids, selected, correction['config']['selected'], correction['model'])
        bundle = joblib.load(BASE/'training_v2/weights/multiview_candidate_ranker.joblib')
        rankings = retrieval.predict(data, retrieval.features(data), data.dev_ids,
                                     bundle['retrieval_model'], bundle['config']['retrieval'])
        return {'grounding': boxes, 'rankings': rankings}
    if method == 'v13_pairwise':
        visual = torch.load(BASE/'training_v13/features/pairwise_lora_dev_regions.pt', weights_only=True)
        config = read('training_v13/results/frozen_config.json')['selected']
        boxes = chosen_boxes(data, data.dev_ids,
                             combined_scores(data, data.dev_ids, visual), config['scale'])
        bundle = joblib.load(BASE/'training_v2/weights/multiview_candidate_ranker.joblib')
        rankings = retrieval.predict(data, retrieval.features(data), data.dev_ids,
                                     bundle['retrieval_model'], bundle['config']['retrieval'])
        return {'grounding': boxes, 'rankings': rankings}
    if method == 'v17_listwise':
        visual = torch.load(BASE/'training_v17/features/listwise_dev_regions.pt', weights_only=True)
        config = read('training_v17/results/frozen_config.json')['selected']
        boxes = chosen_boxes(data, data.dev_ids,
                             combined_scores(data, data.dev_ids, visual), config['scale'])
        bundle = joblib.load(BASE/'training_v2/weights/multiview_candidate_ranker.joblib')
        rankings = retrieval.predict(data, retrieval.features(data), data.dev_ids,
                                     bundle['retrieval_model'], bundle['config']['retrieval'])
        return {'grounding': boxes, 'rankings': rankings}
    version, ablation = method.split(':')
    if version == 'v4':
        sys.path.insert(0, str(BASE/'training_v4/code'))
        import train_fusion as fusion
        features = fusion.ComplementaryFeatures()
        values = fusion.retrieval_features(features.data)
        checkpoint = BASE/'training_v4/weights/complementary_fusion.joblib'
    elif version in ('v5', 'v6'):
        module_name = 'train_activation' if version == 'v5' else 'train_attributes'
        sys.path.insert(0, str(BASE/f'training_{version}/code'))
        module = __import__(module_name)
        features, values = module.prepare()
        fusion = module.fusion
        filename = 'activation_fusion.joblib' if version == 'v5' else 'attribute_fusion.joblib'
        checkpoint = BASE/f'training_{version}/weights'/filename
    else:
        raise ValueError(f'Unrecognized retained method: {method}')
    bundle = joblib.load(checkpoint)
    data = features.data
    config = bundle['config']
    boxes = fusion.select_boxes(features, data.dev_ids,
        fusion.candidate_scores(features, data.dev_ids, bundle['old_grounding'],
                                config['old_grounding'], bundle['grounding']), config['grounding'])
    if ablation == 'retrieval_only':
        boxes = lora_boxes(data)
    rankings = retrieval.predict(data, values, data.dev_ids, bundle['retrieval'], config['retrieval'])
    return {'grounding': boxes, 'rankings': rankings}


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    data = Data()
    rows = []

    def add(name, report, source, method=None):
        counts, n = report['counts'], report['queries_per_task']
        assert report['images'] == 16 and n == 27
        expected = (counts['R@1']+2*counts['R@5']+6*counts['Acc@0.5']+counts['Acc@0.75'])/(10*n)
        assert math.isclose(expected, report['score_out_of_1'], abs_tol=1e-12)
        assert all(0 <= counts[key] <= n for key in counts)
        rows.append({**report, 'name': name, 'source': source, 'method': method})

    initial = read('results/score_round.json')['results']['four_frameworks_equal']
    add('首轮固定五框架融合', initial, 'results/score_round.json')
    add('v1 监督融合头', read('training_v1/results/internal_dev_score.json'), 'training_v1/results/internal_dev_score.json')
    add('v1 RemoteCLIP 视觉 LoRA', read('training_v1/results/visual_lora_internal_dev_score.json'),
        'training_v1/results/visual_lora_internal_dev_score.json', 'v1_lora')
    v2 = read('training_v2/results/ablations.json')
    add('v2 多视角、配对、属性文本全组合', v2['combined'], 'training_v2/results/ablations.json')
    add('v2 新检索 + v1 监督定位', v2['new_retrieval_only'], 'training_v2/results/ablations.json')
    add('v2 新检索 + v1 LoRA 定位（保留）', read('training_v2/results/retained_result.json'),
        'training_v2/results/retained_result.json', 'v2_retained')
    add('v3 区域排序与框回归候选', read('training_v3/results/internal_dev_score.json'), 'training_v3/results/internal_dev_score.json')
    for version, name in (('v4', '中文 CLIP 与语义相容监督'), ('v5', 'GeoRSCLIP QuickGELU 对照'), ('v6', '独立飞机属性证据')):
        path = f'training_{version}/results/internal_dev_score.json'
        report = read(path)
        add(f'{version} {name}全组合', report['combined'], path, version+':combined')
        add(f'{version} 新检索 + 保留 LoRA 定位', report['new_retrieval_only'], path, version+':retrieval_only')
    add('v7 几何修正与同物体框融合', read('training_v7/results/internal_dev_score.json'),
        'training_v7/results/internal_dev_score.json', 'v7_geometry')
    late_rounds = (
        ('v8', 'v8 成对检索', 'combined_with_retained_lora_localization'),
        ('v9', 'v9 扩展 LoRA 候选框', None),
        ('v10', 'v10 Qwen 图库图像比较', 'combined_with_retained_lora_localization'),
        ('v11', 'v11 LoRA 秩与层数搜索', 'combined_with_v2_retrieval'),
        ('v12', 'v12 查询提示交叉验证', 'combined_with_v2_retrieval'))
    for version, name, field in late_rounds:
        path = f'training_{version}/results/internal_dev_score.json'
        report = read(path)
        add(name, report if field is None else report[field], path)
    v13_score_path = BASE/'training_v13/results/internal_dev_score.json'
    if v13_score_path.exists():
        report = read('training_v13/results/internal_dev_score.json')
        add('v13 成对 LoRA + v2 检索', report['combined_with_v2_retrieval'],
            'training_v13/results/internal_dev_score.json', 'v13_pairwise')
    v17_score_path = BASE/'training_v17/results/internal_dev_score.json'
    if v17_score_path.exists():
        report = read('training_v17/results/internal_dev_score.json')
        add('v17 更长训练的候选排序 LoRA', report['combined_with_v2_retrieval'],
            'training_v17/results/internal_dev_score.json', 'v17_listwise')
    # Resolve ties in favor of the earlier, simpler retained method.
    eligible = [row for row in rows if row['method']]
    best = max(eligible, key=lambda row: row['score_out_of_1'])
    predictions = reproduce(best['method'], data)
    reproduced, details = measure(data.cache, data.split['dev'], predictions)
    assert reproduced['counts'] == best['counts']
    assert reproduced['score_out_of_1'] == best['score_out_of_1']
    write_json(OUT/'best_predictions.json', details)
    write_json(OUT/'round_scores.json', rows)
    checked_hashes = {}
    versions = {
        'v2': ('weights/multiview_candidate_ranker.joblib', 'code/train_round.py'),
        'v3': ('weights/region_ranker.pt', 'code/region_ranker.py'),
        'v4': ('weights/complementary_fusion.joblib', 'code/train_fusion.py'),
        'v5': ('weights/activation_fusion.joblib', 'code/train_activation.py'),
        'v6': ('weights/attribute_fusion.joblib', 'code/train_attributes.py'),
        'v7': ('weights/geometry_correction.joblib', 'code/refine_boxes.py'),
        'v8': ('weights/pairwise_retriever.joblib', 'code/pairwise_retrieval.py'),
        'v10': ('weights/qwen_gallery_ranker.joblib', 'code/train_gallery_ranker.py'),
        'v11': ('weights/RemoteCLIP_visual_lora_cvselected.pt', 'code/cv_lora.py'),
        'v12': ('weights/RemoteCLIP_visual_lora_prompt_cv.pt', 'code/lora_prompt_cv.py')}
    for version, (filename, main_code) in versions.items():
        directory = BASE/f'training_{version}'
        report = read(f'training_{version}/results/internal_dev_score.json')
        frozen = read(f'training_{version}/results/frozen_config.json')
        actual = digest(directory/filename)
        assert actual == report['checkpoint_sha256'], version
        code_hashes = frozen['code_sha256']
        if isinstance(code_hashes, str):
            assert code_hashes == digest(directory/main_code), version
        else:
            for path, expected in code_hashes.items():
                source = directory/'code'/path if version == 'v2' else BASE/path
                assert expected == digest(source), str(source)
        assert datetime.fromisoformat(frozen['frozen_at_utc']) <= datetime.fromisoformat(report['scored_at_utc'])
        checked_hashes[f'training_{version}/{filename}'] = actual
    for path, expected in read('training_v2/results/retained_result.json')['required_checkpoints'].items():
        assert digest(BASE/path) == expected
        checked_hashes[path] = expected
    v9 = read('training_v9/results/internal_dev_score.json')
    assert digest(BASE/'training_v2/features/candidates.pt') == v9['config']['candidate_feature_sha256']
    assert digest(BASE/'training_v1/weights/RemoteCLIP_visual_lora.pt') == v9['config']['lora_weight_sha256']
    v13_trials = read('training_v13/results/cv_trials.json')
    v13_frozen = read('training_v13/results/frozen_config.json')
    v13_code = BASE/'training_v13/code/pairwise_lora.py'
    assert v13_frozen['code_sha256'] == digest(v13_code)
    assert len(v13_trials) == 12 and all(row['method'] == 'pairwise_lora' for row in v13_trials)
    assert all(row['fold_queries'] == [21, 17, 16] for row in v13_trials)
    selection_path = BASE/'training_v13/results/selection.json'
    if selection_path.exists():
        selection = read('training_v13/results/selection.json')
        assert selection['decision'] == 'retain_current_best'
        assert max(row['cv_grounding'] for row in v13_trials) <= selection['fit_only_cv_reference']['cv_grounding'] + 1e-9
        assert selection['retained_score_recomputed_in_this_round'] is False
        assert selection['retained_development_score_out_of_1'] == read('optimization_results/current_best.json')['score_out_of_1']
    v13_checkpoint = BASE/'training_v13/weights/RemoteCLIP_visual_lora_pairwise.pt'
    if v13_score_path.exists():
        report = read('training_v13/results/internal_dev_score.json')
        actual = digest(v13_checkpoint)
        assert actual == report['checkpoint_sha256']
        checked_hashes['training_v13/weights/RemoteCLIP_visual_lora_pairwise.pt'] = actual
    checked_hashes['training_v13/code/pairwise_lora.py'] = digest(v13_code)
    cv_rows = []
    fit_only = [
        ('v13', '成对 LoRA 损失', 'pairwise_lora.py', 'fit_only_cv_reference', 'retained_score_recomputed_in_this_round'),
        ('v14', '非线性候选排序', 'tree_ranker.py', None, 'development_score_recomputed'),
        ('v15', 'DINOv2 + RemoteCLIP 直接融合', 'dino_ranker.py', None, 'development_score_recomputed'),
        ('v16', '查询条件化 DINOv2 + CLIP', 'conditional_kernel.py', None, 'development_score_recomputed')]
    if (BASE/'training_v17/results/selection.json').exists():
        fit_only.append(('v17', '更长训练的候选排序 LoRA', 'listwise_lora.py', None, 'development_score_recomputed'))
    for version, name, script, baseline_field, evaluated_field in fit_only:
        directory = BASE/f'training_{version}'
        trials = read(f'training_{version}/results/cv_trials.json')
        frozen = read(f'training_{version}/results/frozen_config.json')
        hashes = frozen['code_sha256']
        if isinstance(hashes, str):
            assert hashes == digest(directory/'code'/script)
            checked_hashes[f'training_{version}/code/{script}'] = hashes
        else:
            for source, expected in hashes.items():
                assert digest(directory/'code'/source) == expected
                checked_hashes[f'training_{version}/code/{source}'] = expected
        for trial in trials:
            assert trial['fold_queries'] == [21, 17, 16]
            assert all(0 <= score <= .7+1e-9 for score in trial['fold_scores'])
            expected = sum(score*n for score, n in zip(trial['fold_scores'], trial['fold_queries']))/sum(trial['fold_queries'])
            assert math.isclose(expected, trial['cv_grounding'], abs_tol=1e-12)
        selection = read(f'training_{version}/results/selection.json')
        assert selection['decision'] == 'retain_current_best'
        assert selection[evaluated_field] is False
        baseline = (selection[baseline_field]['cv_grounding'] if baseline_field else selection['fit_only_baseline'])
        maximum = max(trial['cv_grounding'] for trial in trials)
        assert maximum <= baseline+1e-9
        cv_rows.append({'round': version, 'name': name,
                        'candidate_configurations': len(trials),
                        'fit_only_cv_grounding_max_out_of_0_7': maximum,
                        'promotion_baseline_out_of_0_7': baseline,
                        'internal_dev_evaluated': False,
                        'development_total_out_of_1': reproduced['score_out_of_1'],
                        'development_total_source': 'retained prior best, not a new evaluation',
                        'selection_source': f'training_{version}/results/selection.json'})
    write_json(OUT/'fit_only_cv_rounds.json', cv_rows)
    manifest = read('training_v15/results/dino_features_manifest.json')
    assert manifest['weights_sha256'] == digest(BASE/'weights/DINOv2-base/model.safetensors')
    assert manifest['weights_sha256'] == 'd73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841'
    assert manifest['features_sha256'] == digest(BASE/'training_v15/features/dinov2_region_features.pt')
    assert manifest['code_sha256'] == digest(BASE/'training_v15/code/encode_dino_regions.py')
    v16 = read('training_v16/results/frozen_config.json')
    assert v16['source_feature_sha256'] == manifest['features_sha256']
    assert v16['oof_lora_sha256'] == digest(BASE/'training_v15/features/remoteclip_lora_oof_scores.pt')
    checked_hashes['weights/DINOv2-base/model.safetensors'] = manifest['weights_sha256']
    v18_manifest = read('training_v18/results/siglip2_features_manifest.json')
    v18_frozen = read('training_v18/results/frozen_config.json')
    v18_trials = read('training_v18/results/cv_trials.json')
    v18_selection = read('training_v18/results/selection.json')
    assert v18_selection['decision'] == 'retain_current_best'
    assert v18_selection['development_score_recomputed'] is False
    assert len(v18_trials['grounding']) == 456 and len(v18_trials['retrieval']) == 35
    for task, field in (('grounding', 'cv_grounding'), ('retrieval', 'cv_retrieval')):
        for trial in v18_trials[task]:
            assert trial['fold_queries'] == [21, 17, 16]
            expected = sum(score*n for score, n in zip(trial['fold_scores'], trial['fold_queries']))/54
            assert math.isclose(expected, trial[field], abs_tol=1e-12)
    g_max = max(row['cv_grounding'] for row in v18_trials['grounding'])
    r_max = max(row['cv_retrieval'] for row in v18_trials['retrieval'])
    assert g_max <= v18_frozen['grounding']['cv_grounding']+1e-9
    assert r_max <= v18_frozen['retrieval_fit_only_baseline']+1e-9
    for source, expected in (
        ('training_v18/code/encode_siglip2.py', v18_manifest['code_sha256']),
        ('training_v18/code/train_siglip2.py', v18_frozen['code_sha256']),
        ('training_v18/features/siglip2_train_features.pt', v18_manifest['features_sha256']),
        ('weights/SigLIP2-base-512/model.safetensors', v18_manifest['weights_sha256'])):
        assert digest(BASE/source) == expected
        checked_hashes[source] = expected
    assert v18_frozen['feature_sha256'] == v18_manifest['features_sha256']
    assert v18_manifest['weights_sha256'] == 'fe0e601c625e69eed8e73500d39e9b6164403fe03db8048e87913c3cefbbb3fe'
    cv_rows.append({'round': 'v18', 'name': 'SigLIP 2 定位与检索融合',
                    'candidate_configurations': 491,
                    'fit_only_cv_grounding_max_out_of_0_7': g_max,
                    'promotion_baseline_out_of_0_7': v18_frozen['grounding']['cv_grounding'],
                    'fit_only_cv_retrieval_max_out_of_0_3': r_max,
                    'retrieval_promotion_baseline_out_of_0_3': v18_frozen['retrieval_fit_only_baseline'],
                    'internal_dev_evaluated': False,
                    'development_total_out_of_1': reproduced['score_out_of_1'],
                    'development_total_source': 'retained prior best, not a new evaluation',
                    'selection_source': 'training_v18/results/selection.json'})
    write_json(OUT/'fit_only_cv_rounds.json', cv_rows)
    if v17_score_path.exists():
        frozen = read('training_v17/results/frozen_config.json')
        report = read('training_v17/results/internal_dev_score.json')
        trials = read('training_v17/results/cv_trials.json')
        assert frozen['code_sha256'] == digest(BASE/'training_v17/code/listwise_lora.py')
        assert frozen['selected']['cv_grounding'] > frozen['fit_only_baseline']+1e-9
        assert len(trials) == 12
        for trial in trials:
            assert trial['fold_queries'] == [21, 17, 16]
            expected = sum(score*n for score, n in zip(trial['fold_scores'], trial['fold_queries']))/54
            assert math.isclose(expected, trial['cv_grounding'], abs_tol=1e-12)
        checkpoint = BASE/'training_v17/weights/RemoteCLIP_listwise_lora.pt'
        assert digest(checkpoint) == report['checkpoint_sha256']
        assert datetime.fromisoformat(frozen['frozen_at_utc']) <= datetime.fromisoformat(report['scored_at_utc'])
        checked_hashes['training_v17/code/listwise_lora.py'] = frozen['code_sha256']
        checked_hashes['training_v17/weights/RemoteCLIP_listwise_lora.pt'] = report['checkpoint_sha256']
    archive = TrainArchive()
    with ZipFile(BASE/'data/original_train_only.zip') as zipped:
        assert set(zipped.namelist()) == archive.allowed
        assert len(zipped.namelist()) == 98
    fit, dev = set(data.split['fit']), set(data.split['dev'])
    assert not fit & dev and len(fit) == 33 and len(dev) == 16
    assert all(set(group) <= fit or set(group) <= dev for group in data.split['overlap_groups'])
    folds = data.folds()
    assert sorted(sum(folds, [])) == sorted(data.fit_ids)
    audits = []
    for path in sorted(BASE.glob('training_v*/results/*data_access.json')):
        audit = json.loads(path.read_text(encoding='utf-8'))
        assert audit['all_accesses_within_training_whitelist']
        assert not audit['holdout_or_test_entries_accessed']
        assert set(audit['accessed_entries']) <= archive.allowed
        audits.append(str(path.relative_to(BASE)))
    archive.close()
    verification = {'score_arithmetic_for_all_rounds_verified': True,
                    'best_result_reproduced_from_models_and_cached_inference_features': True,
                    'original_train_only_archive_entries': 98,
                    'fit_and_dev_overlap_groups_disjoint': True,
                    'official_validation_or_test_used': False,
                    'verified_data_access_audits': audits,
                    'checkpoint_sha256_and_recorded_code_verified': checked_hashes,
                    'v13_fit_only_cv_trials_verified': len(v13_trials),
                    'v13_development_score_recomputed': v13_score_path.exists(),
                    'fit_only_cv_rounds_verified': cv_rows,
                    'dino_pretrained_weights_and_features_verified': True,
                    'scope': 'iterative development; the internal-dev score has informed model retention',
                    'verified_at_utc': datetime.now(timezone.utc).isoformat()}
    write_json(OUT/'verification.json', verification)
    retained = {'method': best['method'], 'name': best['name'], **reproduced,
                'target_score': .6, 'target_met': reproduced['score_out_of_1'] >= .6,
                'official_validation_or_test_used': False,
                'selection_consulted_internal_dev': True,
                'model_export_or_official_submission_image_created': False,
                'source_report': best['source'], 'verification': 'verification.json'}
    write_json(OUT/'current_best.json', retained)
    table = ['| 轮次 / 方案 | R@1 | R@5 | Acc@0.5 | Acc@0.75 | 总分 / 1 |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        counts = row['counts']
        cells = [f'{counts[key]}/27' for key in ('R@1', 'R@5', 'Acc@0.5', 'Acc@0.75')]
        table.append('| '+row['name']+' | '+' | '.join(cells)+f' | {row["score_out_of_1"]:.4f} |')
    cv_table = ['| 轮次 | 最佳拟合折定位加权分 / 0.7 | 晋级基线 / 0.7 | 当前沿用开发总分 / 1 |',
                '|---|---:|---:|---:|']
    for row in cv_rows:
        cv_table.append(f"| {row['round']} {row['name']} | {row['fit_only_cv_grounding_max_out_of_0_7']:.4f} | "
                        f"{row['promotion_baseline_out_of_0_7']:.4f} | {row['development_total_out_of_1']:.4f}（沿用） |")
    status = '已达到' if retained['target_met'] else '尚未达到'
    text = f'''# 连续优化记录与当前保留结果

当前最佳为 **{retained['score_out_of_1']:.4f}/1**，方案：{best['name']}；{status} 0.6 目标。所有结果采用同一 16 张原始训练内部开发图、每任务 27 条查询计算，不能称为官方测试成绩。

总分 = 0.1×R@1 + 0.2×R@5 + 0.6×Acc@0.5 + 0.1×Acc@0.75。

{chr(10).join(table)}

未晋级的轮次只完成拟合折筛选；下表总分沿用已核验的既有最佳，不是新增评测。定位加权分只计算 0.6×Acc@0.5 + 0.1×Acc@0.75，最高为 0.7。

{chr(10).join(cv_table)}

## 数据与模型使用

源数据仅为本地隔离的 49 张公开原始 train 民航图。33 张图用于拟合，16 张用于内部开发评分；SIFT 重叠场景不跨界。各轮超参数使用 33 张内的场景分组三折选择，模型和参数在开发评分前保存。开发分数被反复用于架构迭代和保留方案，因此这里是开发成绩。折内较小图库的交叉验证分数只用于选择参数，不替代本表。

模型输入来自查询文本、图像与预测框。没有使用官方验证集、测试集或旧 47 张留出图；没有按文件名、查询 ID 或真值编写预测。独立属性分类只读取预测区域，不把查询或答案送进属性提示。检索的属性聚合覆盖图库中每张图的检测候选，没有使用定位任务提供的正确图像映射。

实际设备为本机 RTX 4060 Laptop 8 GB。大视觉语言模型使用 NF4 4 位量化，推理进程通过审计钩子阻止联网；本轮没有声称已在 A100 上重跑。大模型顺序加载。

## 每轮改动与复现

1. `training_v2`：GeoGround 旋转框、Qwen 双向候选比较、实际输出框的区域编码、文本属性提示、Qwen 分数去饱和。依次运行 `code/geoground_views.py`、`code/qwen_pairs.py`、`code/encode_candidates.py`、`code/encode_prompts.py`、`code/train_round.py`、`code/retain.py`。
2. `training_v3`：冻结视觉向量与查询交互、查询内中心化排序、候选差分训练、框回归。运行 `code/region_ranker.py`。三折未选择视觉交互或框回归，保留了完整负结果。
3. `training_v4`：中文 CLIP 证据、原始/扩展候选对照、已训练头与新头的折外融合、按查询属性构造语义相容监督。运行 `code/train_fusion.py`。
4. `training_v5`：GeoRSCLIP 的 QuickGELU 激活函数对照，图像和文本同步重新编码。先运行 `code/encode_activation.py`，再运行 `code/train_activation.py`。官方示例不足以断言 L/336 先前的激活配置错误，因此只作为实验对照报告。
5. `training_v6`：在预测区域上独立提取发动机数、涂装、飞机完整度；先运行 `code/extract_attributes.py`，再运行 `code/train_attributes.py`。同一物体只缓存预测证据，没有缓存标签答案。
6. `training_v7`：用拟合图学习紧凑的框几何修正，并比较同一物体的多模型框融合。运行 `code/refine_boxes.py`。校准只在拟合图的场景分组折内训练，开发图仍用已保存的 LoRA 选框器。
7. `training_v8`：成对比较检索候选并用折内检索指标筛选，运行 `code/pairwise_retrieval.py`；开发总分为 0.5111。
8. `training_v9`：扩展 LoRA 定位候选框，运行 `code/score_expanded_lora.py`；开发总分为 0.5074。
9. `training_v10`：比较 Qwen 图库图像证据，运行 `code/encode_gallery_pairs.py`、`code/train_gallery_ranker.py` 和 `code/test_gallery_ranker.py`。1,204 组成对比较已归档并通过访问检查；最终总分与现有最佳相同，为 0.5259。
10. `training_v11`：在 33 张拟合图内搜索 LoRA 秩和解冻层数，运行 `code/cv_lora.py`、`code/score_lora_cv.py`；没有超过现有开发总分。
11. `training_v12`：在拟合折内比较查询提示模式，运行 `code/lora_prompt_cv.py`；没有超过现有开发总分。
12. `training_v13`：在预测框上用图文二元对齐与成对硬负例排序混合训练 LoRA，运行 `code/pairwise_lora.py`。四档成对损失权重、三折场景分组交叉验证的最佳加权定位分未超过 0.4593 基线，因此没有重评内部开发集或保存替代权重；详情见 `training_v13/results/selection.json`。
13. `training_v14`：在相同候选池上使用 ExtraTrees、HistGradientBoosting 预测 IoU/定位效用；三折最佳 0.4741，未超过已复现的 v3 场景折定位基线 0.4852。运行 `code/tree_ranker.py`。
14. `training_v15`：下载 Apache-2.0 的 `facebook/dinov2-base`，在 518×518 的紧框与 1.5 倍上下文上提取 CLS 和平均 patch 特征，再与 RemoteCLIP 折内适配器结合。运行 `code/encode_dino_regions.py` 与 `code/dino_ranker.py`；三折最佳 0.4722，未晋级。
15. `training_v16`：将 DINOv2 区域外观核与 CLIP 查询语义核相乘，让排序依赖查询文本；运行 `code/conditional_kernel.py`。三折最佳 0.4833，未超过 0.4852 基线，保留原方案。
16. `training_v17`：固定候选池，使用查询内定位效用分布训练候选排序 LoRA；学习率 0.0005/0.002，在 6/12 个 epoch 比较折外定位分。运行 `code/listwise_lora.py`；最佳 0.4315，未超过 0.4852 拟合折基线，未进入开发集评分。
17. `training_v18`：对照多语言 SigLIP 2 的中文、英文及融合文本特征，分别比较候选框排序与全图库检索。运行 `code/encode_siglip2.py`、`code/train_siglip2.py`。定位最佳 0.4611/0.7，检索最佳 0.1630/0.3，均未超过拟合折基线；没有新增开发评分。

在项目根目录使用 `fusion_train_only/.venv/Scripts/python.exe` 调用上述完整脚本路径。实验依赖先前的预训练特征和模型文件；此处是实验复现入口，未作为接收任意新图像的正式提交接口。

## 结果和权重

`round_scores.json` 保存首轮和 v1–v12 的同协议开发总分和命中数。`fit_only_cv_rounds.json` 保存 v13–v18 的筛选分数与沿用总分。`current_best.json` 是当前结果索引，`best_predictions.json` 保存逐条开发预测，`verification.json` 记录分数验算、原始训练图白名单、场景分离和代码/权重哈希检查。v13–v18 没有独立开发成绩，因为它们没有通过拟合折筛选。

新增 DINOv2 预训练权重存放于 `weights/DINOv2-base/model.safetensors`（346,345,912 字节），固定版本 `f9e44c814b77203eaa57a6bdbbd535f21ede1415`，SHA-256 `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`。这是通用预训练权重；本轮没有生成通过 0.6 分门槛的融合权重或正式镜像包。模型来源、特征哈希及数据范围见 `training_v15/results/dino_features_manifest.json`。

新增 SigLIP 2 预训练权重存放于 `weights/SigLIP2-base-512/model.safetensors`（1,503,344,520 字节），固定版本 `a89f5c5093f902bf39d3cd4d81d2c09867f0724b`，SHA-256 `fe0e601c625e69eed8e73500d39e9b6164403fe03db8048e87913c3cefbbb3fe`。模型和 tokenizer 已完成本机离线加载与原始训练图特征提取；这不代表正式 A100 容器实测。

拟合数据诊断：54 条查询只包含 18 种不同文字，4 个图像—文字组合含多个目标。同一文字固定一个排序时，33 张拟合图库的 R@1 上限为 0.3889；同一图像—文字组合只能输出一个框时，当前候选池可实现的拟合定位加权上限约 0.6352。这里的上限通过拟合标签事后计算，只解释数据歧义，不属于模型得分，也不用于输出预测。详情见 `fit_only_diagnostics.json`。

当前保留方案的检查点从 `{best['source']}` 追溯。若方法为 `v2_retained`，使用 `training_v2/weights/multiview_candidate_ranker.joblib` 内的检索头和 `training_v1/weights/RemoteCLIP_visual_lora.pt`，并通过 `training_v2/code/retain.py` 复现；它们还依赖上一级 `weights/` 中的预训练网络。

实验参数已保存，正式权重导出包和 Docker 镜像包未生成。验算脚本为 `optimization_results/code/finalize.py`。
'''
    (OUT/'README.md').write_text(text, encoding='utf-8')
    (OUT/'round_scores.csv').write_text('round,R@1_hits,R@5_hits,Acc0.5_hits,Acc0.75_hits,queries,score_out_of_1\n'+
        '\n'.join(','.join([row['name']]+[str(row['counts'][key]) for key in ('R@1', 'R@5', 'Acc@0.5', 'Acc@0.75')]+
                          ['27', str(row['score_out_of_1'])]) for row in rows)+'\n', encoding='utf-8-sig')
    (OUT/'fit_only_cv_rounds.csv').write_text(
        'round,fit_only_cv_grounding_max_out_of_0_7,promotion_baseline_out_of_0_7,internal_dev_evaluated,development_total_out_of_1,development_total_source\n'+
        '\n'.join(','.join([row['round'], str(row['fit_only_cv_grounding_max_out_of_0_7']),
                            str(row['promotion_baseline_out_of_0_7']), 'false',
                            str(row['development_total_out_of_1']), 'retained prior best']) for row in cv_rows)+'\n',
        encoding='utf-8-sig')
    print('Verified retained result:', retained, flush=True)


if __name__ == '__main__':
    main()
