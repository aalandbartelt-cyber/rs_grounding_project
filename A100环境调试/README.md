# 自强 5000 · A100 环境调试与首轮实测

这里保存首次在自强 5000 的 A100 上跑通民航检索与定位时使用的脚本、操作步骤和原始指标。最终模型组合及 Docker 成果见 [11.19](../11.19/README.md)。

## 1. 申请 A100 并准备 Python 环境

选择一张 A100。在 Notebook 的终端执行 `nvidia-smi`，确认系统识别出 GPU。首轮使用现成 CUDA 环境，成功启动时 PyTorch 为 `2.6.0+cu124`、Transformers 为 `4.51.3`，`torch.cuda.is_available()` 返回 `True`。下文默认 Python 在 `/tmp/rsg-env/bin/python`，也可以将 `RSG_PYTHON` 改为当前可用的虚拟环境解释器。

```bash
PY=/tmp/rsg-env/bin/python
"$PY" -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

如果容器重启后 `/tmp/rsg-env` 消失，可先用当前已能识别 CUDA 的系统 Python 重建环境，再安装 `code/requirements.txt` 中的应用依赖。保留容器提供的 CUDA 版 PyTorch。

```bash
python -m venv --system-site-packages /tmp/rsg-env
/tmp/rsg-env/bin/python -m pip install -r /实际路径/A100环境调试/code/requirements.txt
```

## 2. 解包并核验数据与权重

使用首次试跑时准备的 `rs_zq5000_fast_preflight.zip` 与 `rs_zq5000_model_weights.zip`。前者提供人工复核的民航子集，后者提供两个模型权重。将两包上传至 `/home/jovyan`。GitHub 保存脚本与指标，复现时使用本地已有的两份归档。

```bash
WORK="$HOME/rsg_work/fast"
mkdir -p "$WORK"
"$PY" -m zipfile -e "$HOME/rs_zq5000_fast_preflight.zip" "$WORK"
"$PY" -m zipfile -e "$HOME/rs_zq5000_model_weights.zip" "$WORK"
"$PY" "$WORK/code/prepare_check.py" --models all --cuda --reuse-runtime
```

确认显示 `clip: verified`、`dino: verified` 与 `CUDA OK` 后再运行。`--reuse-runtime` 使用 Notebook 已验证的 CUDA 环境。预检脚本还核对复核数据的 SHA-256 与划分范围。

## 3. 先验证 1 张图，再运行全量定位与检索

从仓库中的 `A100环境调试/` 目录运行下面的脚本。脚本按顺序重复预检、跑一张图、评估全部 47 张图、训练和评估检索适配层，最后打印四项指标与总分。每轮在 `rsg_work/fast/runs/` 下建立独立目录。

```bash
bash run_a100_baseline.sh
```

逐步排查时，查看脚本中编号的命令，并在对应运行目录打开 `01_preflight.log` 到 `05_score.log`。第 2 步单图测试只检查流程和输出格式。第 3 步用检测最高分框评分。第 4 步用训练集的 49 张图学习检索适配层，再在 47 张留出图上计分。第 5 步写出 `score_summary.txt` 和 `score_summary.json`，并把结果 ZIP 保存到 `/home/jovyan`，可在左侧文件列表刷新后下载。

## 4. 首轮 A100 正确运行结果

下表来自实际下载的 `rsg_experiments_20261119_132927.tar.gz`。两份指标 JSON 保留原样，均记录 `device: cuda`，在同一批 47 张民航留出图上计算，每项 75 条查询。

| 组合 | R@1 | R@5 | Acc@0.5 | Acc@0.75 | 总分 / 1 |
|---|---:|---:|---:|---:|---:|
| 零样本检索 + 基线定位 | 1/75 | 11/75 | 45/75 | 26/75 | 0.4253 |
| 微调检索 + 基线定位 | 7/75 | 12/75 | 45/75 | 26/75 | **0.4360** |

总分为 `0.1×R@1+0.2×R@5+0.6×Acc@0.5+0.1×Acc@0.75`。定位脚本计时 12.05 秒，检索脚本计时 4.27 秒。候选框覆盖率分别为 68/75 和 39/75，用来诊断选框与边界误差。

- [定位原始指标](results/a100_grounding.json)
- [检索原始指标](results/a100_retrieval.json)
- [赛题计分汇总](results/score_summary.txt)
