# 11.19 · 区域匹配方法

本轮在复核民航训练集内部用三折验证选参数，再固定方法评估剩余 47 张民航图。最终采用首轮整图检索与第二轮区域定位。区域检索实验的 R@1 回落，完整对照保存在 [成绩汇总](../results/score_summary.txt)。本轮成绩来自本机 CPU。

`benchmark_regions.py` 训练并评估候选区域方案，`preserve_retrieval.py` 复核首轮整图检索适配层与改进定位的组合。训练预检、首轮模型脚本和 A100 环境记录单独保存在 [A100 环境调试](../../A100环境调试/README.md)。已有复核数据与模型的环境可把本目录两个脚本复制到 `rsg_work/fast/code/` 后运行，先 `tune`，再 `evaluate`，最后执行 `preserve_retrieval.py` 并提供上一轮检索适配层文件。

```bash
PY=/tmp/rsg-env/bin/python
WORK="$HOME/rsg_work/fast"
"$PY" "$WORK/code/benchmark_regions.py" tune
"$PY" "$WORK/code/benchmark_regions.py" evaluate
"$PY" "$WORK/code/preserve_retrieval.py" --device cpu \
  --reference-adapter "$WORK/experiments/reference/adapter.safetensors"
```

需要重现本轮 CPU 记录时，给 `tune` 和 `evaluate` 也传入 `--device cpu`。
