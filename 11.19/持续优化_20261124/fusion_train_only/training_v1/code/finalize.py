"""Verify completed training rounds and write the Chinese experiment record."""
import importlib.metadata
import json
from datetime import datetime,timezone
from zipfile import ZipFile

import torch

from data import BASE,Data
from common import digest,write_json

OUT=BASE/'training_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    data=Data()
    files={'监督融合头':'internal_dev_score.json','非线性候选排序':'tree_internal_dev_score.json',
           '增强视觉适配器候选':'adapter_internal_dev_score.json','全候选检索聚合候选':'retrieval_pool_internal_dev_score.json',
           '拟合图全局系数校准':'calibration_internal_dev_score.json'}
    if (OUT/'results/visual_lora_internal_dev_score.json').exists():
        files['RemoteCLIP 视觉 LoRA 微调']='visual_lora_internal_dev_score.json'
    if (OUT/'results/visual_lora_full_internal_dev_score.json').exists():
        files['视觉 LoRA 同时接入检索与定位']='visual_lora_full_internal_dev_score.json'
    rounds={name:read(OUT/'results'/file) for name,file in files.items()}
    cv=read(OUT/'results/all49_grouped_cv_score.json')
    protocol=read(OUT/'results/all49_cv_protocol.json')
    assert cv['protocol_sha256']==digest(OUT/'results/all49_cv_protocol.json')
    validation_labels=[label for fold in protocol['folds'] for label in fold]
    assert len(validation_labels)==49 and len(set(validation_labels))==49
    assert set(validation_labels)==set(data.split['fit']+data.split['dev'])
    for group in data.split['overlap_groups']:
        assert sum(set(group)<=set(fold) for fold in protocol['folds'])==1
    for i,fold in enumerate(protocol['folds'],1):
        checkpoint=OUT/f'weights/all49_cv_fold_{i}.pt'
        model=torch.load(checkpoint,weights_only=True)
        assert set(model['training_labels']).isdisjoint(fold)
        assert set(model['training_labels'])|set(fold)==set(validation_labels)
        assert digest(checkpoint)==cv['folds'][i-1]['checkpoint_sha256']
    for row in list(rounds.values())+[cv]:
        n=row['queries_per_task'];c=row['counts']
        expected=(c['R@1']+2*c['R@5']+6*c['Acc@0.5']+c['Acc@0.75'])/(10*n)
        assert abs(expected-row['score_out_of_1'])<1e-12
        assert not row['official_validation_or_test_used']
    full=torch.load(OUT/'weights/all49_trained_fusion.pt',weights_only=True)
    assert set(full['training_labels'])==set(validation_labels)
    augmented=torch.load(OUT/'features/augmented_fit_regions.pt',weights_only=True)
    assert len(augmented['jobs'])==432
    assert all(job['image'] in data.fit_ids for job in augmented['jobs'])
    archive_manifest=read(BASE/'data/manifest.json')
    assert digest(BASE/'data/original_train_only.zip')==archive_manifest['archive_sha256']
    assert digest(BASE/'data/pretrained_train_features.pt')==archive_manifest['pretrained_features_sha256']
    with ZipFile(BASE/'data/original_train_only.zip') as archive:
        allowed=set(archive.namelist())
    assert len(allowed)==98 and all('/train/' in entry for entry in allowed)
    logs=sorted((OUT/'results').glob('*data_access.json'))
    for path in logs:
        row=read(path)
        assert row['all_accesses_within_training_whitelist'] and not row['holdout_or_test_entries_accessed']
        assert set(row['accessed_entries'])<=allowed
    weights={path.name:{'bytes':path.stat().st_size,'sha256':digest(path)} for path in (OUT/'weights').iterdir() if path.is_file()}
    lora_verified=None
    if (OUT/'weights/RemoteCLIP_visual_lora.pt').exists():
        lora=torch.load(OUT/'weights/RemoteCLIP_visual_lora.pt',weights_only=True)
        assert set(lora['config']['training_images'])==set(data.split['fit'])
        assert any(torch.count_nonzero(t)>0 for name,t in lora['state'].items() if name.endswith('.b'))
        lora_verified={'trainable_parameters':sum(t.numel() for t in lora['state'].values()),
                       'low_rank_updates_nonzero':True,'trained_on_original_33_fit_only':True}
    versions={name:importlib.metadata.version(name) for name in ('torch','transformers','open_clip_torch','scikit-learn','scipy','numpy','joblib')}
    verification={'verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'all_score_arithmetic_matches':True,'all49_cv_is_out_of_fold':True,
        'all49_cv_groups_preserved':True,'augmentation_used_original_fit_only':True,
        'all_accesses_original_public_train_only':True,'official_validation_or_test_used':False,
        'training_data_sha256_unchanged':True,'data_access_logs':[p.name for p in logs],
        'weight_files':weights,'packages':versions,
        'visual_lora_verification':lora_verified,
        'code_sha256':{p.name:digest(p) for p in (OUT/'code').glob('*') if p.is_file()}}
    write_json(OUT/'results/training_verification.json',verification)
    original=rounds['监督融合头'];n=original['queries_per_task']
    best_name,best=max(rounds.items(),key=lambda item:item[1]['score_out_of_1'])
    write_json(OUT/'results/retained_result.json',{'best_same_16_image_internal_dev_method':best_name,
        'best_same_16_image_internal_dev_score':best['score_out_of_1'],
        'target_score':.6,'target_met_on_same_16_images':best['score_out_of_1']>=.6,
        'all49_grouped_cv_score':cv['score_out_of_1'],
        'official_validation_or_test_used':False,
        'validation_results_used_to_retain_best_development_run':True})
    retained_weights=['RemoteCLIP_visual_lora.pt','RemoteCLIP_lora_retrieval_head.pt','trained_fusion_final.pt']
    write_json(OUT/'results/retained_weights.json',{
        'score_on_same_16_image_internal_dev':best['score_out_of_1'],
        'scope':'development checkpoint; target 0.6 not met' if best['score_out_of_1']<.6 else 'development checkpoint',
        'pretrained_networks_directory':'../../weights',
        'files':{name:weights[name] for name in retained_weights if name in weights},
        'inference_implementation':['code/lora_visual.py','code/lora_retrieval.py'],
        'official_validation_or_test_used':False})
    lines=['# 基于最佳模型组合的监督训练与调参','',
        '使用 RemoteCLIP、GeoRSCLIP、GroundingDINO、GeoGround 与 Qwen3-VL。先冻结基础网络训练候选排序头'
        '和图像检索头，再单独尝试 RemoteCLIP 视觉骨干末两块的 LoRA 微调；逐项记录所有开发轮次。',
        '',f"原先 16 张训练内部验证图的最高分为 **{best['score_out_of_1']:.4f}/1**，"
        f"{'已达到' if best['score_out_of_1']>=.6 else '未达到'} 0.6，方案为{best_name}。"
        '此前同一划分的固定系数融合为 0.4667/1。',
        '', '| 训练/调参方案 | 同一 16 张内部验证图总分 |', '|---|---:|']
    for name,row in rounds.items():
        lines.append(f"| {name} | {row['score_out_of_1']:.4f} |")
    lines.extend(['',f"监督融合头：R@1={original['counts']['R@1']}/{n}，R@5={original['counts']['R@5']}/{n}，"
        f"Acc@0.5={original['counts']['Acc@0.5']}/{n}，Acc@0.75={original['counts']['Acc@0.75']}/{n}。",
        '',f"最终保留方案：R@1={best['counts']['R@1']}/{n}，R@5={best['counts']['R@5']}/{n}，"
        f"Acc@0.5={best['counts']['Acc@0.5']}/{n}，Acc@0.75={best['counts']['Acc@0.75']}/{n}。"
        '采用完整 LoRA 集成时需加载 retained_weights.json 中列明的适配器与检索头。',
        '', '总分始终使用 0.1×R@1 + 0.2×R@5 + 0.6×Acc@0.5 + 0.1×Acc@0.75，满分 1。',
        '', '第一阶段只用 33 张拟合图训练和选择参数，使用场景分组三折交叉验证。'
        '这部分折内图库比 16 张验证图库更小，0.6407 的交叉验证分数不能替代 0.5000 的固定划分成绩。'
        '后续架构尝试参考了内部验证表现，所有结果均属于开发结果。',
        '', '额外的非线性模型、反向传播排序网络、框回归、432 个旋转/翻转/亮度增强样本适配器，以及'
        '完整候选集合聚合均已实际运行。未改善交叉验证的候选保留原模型，未伪装成新的有效权重。',
        '', f"为检查场景划分的影响，又对全部 49 张原始训练图做三折场景分组交叉验证："
        f"**{cv['score_out_of_1']:.4f}/1**，共 {cv['queries_per_task']} 条检索查询和同数定位查询。",
        '', '| 折 | 拟合图 | 验证图库 | 查询数/任务 | 总分 |', '|---|---:|---:|---:|---:|'])
    for fold in cv['folds']:
        lines.append(f"| {fold['fold']} | {fold['training_images']} | {fold['validation_images']} | {fold['queries_per_task']} | {fold['score_out_of_1']:.4f} |")
    lines.extend(['', '三折结果按全部查询的命中数汇总，没有挑选最高的一折。每张图只在未使用其训练的模型上计分，'
        '重叠场景保持在同一折。原先 16 张内部验证图在这项实验中轮流进入拟合部分，因此它与固定 16 张实验采用不同协议。'
        '框架和参数此前已通过这批训练数据进行开发，这项交叉验证也不能视为未见过的最终测试成绩。',
        '', '两个大模型以 NF4 4 位量化在本机 RTX 4060 Laptop 上运行，推理进程阻止联网。'
        '数据源只包含公开 train 下已复核的 49 张民航图及其标注，没有读取官方验证集或测试集。'
        '推理接口使用查询、图像和检测候选；训练真值只进入训练损失与评分函数。',
        '', '当前可复用权重：`weights/trained_fusion_final.pt` 是 33 图拟合、固定 16 图验证的监督融合头。'
        '`weights/all49_trained_fusion.pt` 是完成交叉验证后在全部 49 张公开训练图上拟合的融合头，'
        '它的性能由独立保存的折外模型预测估计，没有用它自己的拟合图计算成绩。'
        '这些是特征之上的共享模型参数，需配合上一级目录的预训练网络；并非微调后的完整视觉骨干。',
        '', '若存在 `weights/RemoteCLIP_visual_lora.pt`，该文件保存实际更新的视觉低秩参数，'
        '需通过 `code/lora_visual.py` 的参数化结构加载到官方 RemoteCLIP 权重上。'
        '训练损失、训练内部的选轮依据和固定 16 图成绩分别见 visual_lora_training_history、'
        'visual_lora_inner_selection 和 visual_lora_internal_dev_score。',
        '', '文件索引：`results/internal_dev_score.json`、`results/all49_grouped_cv_score.json`、'
        '`results/all49_cv_protocol.json`、`results/training_verification.json`。'
        '日志、每条预测、所有候选的交叉验证试验均保留在 results。',
        '', '未创建正式提交镜像包。'])
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f"Verified training records. Best fixed-dev={best['score_out_of_1']:.4f}, all49 grouped-CV={cv['score_out_of_1']:.4f}.",flush=True)


if __name__=='__main__':main()
