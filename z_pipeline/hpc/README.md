# 自强5000：本项目的超算迁移方法

核查日期：2026-11-21。只做使用方法整理和迁移准备，尚未登录账号、申请资源、消耗机时或启动远程训练。本机大依赖安装不再继续；已有数据导出保留。

## 1. 找到的资源与适用性

用户保存的《上海大学自强5000超级计算平台GPU计算资源实例介绍》列出：单张 A100 40GB、4 vCPU、40Gi 系统内存的 PyTorch 实例，以及 2/4/8 卡配置；共享存储为 500GB。上述是手册配置，不代表当前账户配额或实时可用配置。

当前官网 https://hpc5000.shu.edu.cn/ 公开列出 CPU、A100、A800、4090D 等资源入口。以登录后本人账号实际展示的资源、权限、价格和配额为准。

我们的首选是已经获授权的单卡 NVIDIA GPU PyTorch 容器。手册中的 A100 40GB 显存容量约为本机 RTX4060 Laptop 8GB 的五倍，但这不是速度倍数。实际每步速度、数据加载、资源等待和费用需要测量。不要仅为小数据集第一轮实验就申请多卡。

远程训练把模型、数据 batch、优化器状态的主要内存/显存负担移到服务器；本机只保留代码编辑、文件上传、日志和结果查看。采用合适的后台运行方式后，训练不依赖本机持续执行 Python。

## 2. 先区分 CPU 集群与 GPU 容器

- 《自强5000快速使用手册》中的 login、debug、queue1 等属于 CPU 集群使用流程。登录节点不可直接运行计算任务，需 sbatch 提交到计算节点。
- GPU 案例手册说明的是：申请 GPU 容器，进入实例 Terminal，上传项目并运行。不要把 CPU 队列的免费 debug 规则套用为免费 GPU，也不要把 CPU Slurm 脚本直接当成 GPU 申请脚本。
- 若账号展示的是另一套 GPU 作业调度入口，应先核实该入口的 GPU 分区与申请方式，再编写作业脚本，不猜测分区名和 sbatch 参数。

## 3. 登录与权限

1. 打开 https://hpc5000.shu.edu.cn/ ，进入平台并使用学校统一身份认证。
2. 查看账户状态、可用资源、余额/机时及 GPU 实例。手册规定学生使用的实例由教师授权，若没有实例，先请导师检查是否已分配 GPU 权限。
3. 手册中的校内 SSH 登录需要 VPN2；GPU 网页入口是否也需要校园网络，按当前平台实际要求处理。
4. 有现成且获授权的实例，先检查状态即可；需要新建时先核对当前计费规则。本文不代替授权申请实例，也不提供任何凭证。

## 4. 资源与存储的实际检查

进入已授权 GPU 实例的 Terminal 后，先运行：

```bash
nvidia-smi
python --version
conda env list
pwd
df -h "$HOME"
```

检查实际 GPU 型号、可见显存与使用情况；再用下方 probe_runtime.py 验证 PyTorch 是否真正能调用 CUDA。nvidia-smi 的 CUDA Version 是驱动所支持的 CUDA 版本信息，不等于当前 PyTorch 编译使用的 CUDA 版本。

手册建议所有文件放在用户持久化目录中：代码、数据、模型缓存、Conda 环境、检查点和日志均应在确认持久化的挂载目录。不要把唯一的权重放在 /tmp 或未经确认的容器临时目录。

建议布局（先核实 $HOME 的持久化属性）：

```text
$HOME/rs_grounding_z/code/          本次独立代码
$HOME/rs_grounding_z/datasets/      原始训练包
$HOME/rs_grounding_z/generated/     在 Linux 重新生成的数据索引与评测
$HOME/rs_grounding_z/models/        固定版本的模型权重
$HOME/rs_grounding_z/runs/          每个实验的日志与结果
$HOME/rs_grounding_z/cache/         模型及依赖缓存
```

## 5. 迁移哪些文件

- 上传原始 rs_grounding_train_v1 数据目录及本次 z_evaluation/、z_pipeline/、z_research/ 代码目录，保持原始训练包文件内容不变。
- 当前新增代码尚未提交上传到 GitHub，直接 clone 远程 Z 不能取得这些未提交内容；可用本次本地迁移包上传。
- 不上传 Windows venv、pip 缓存或整套本地安装。它们不能直接作为 Linux 环境使用。
- 不把本地 data_v2 的 Windows 绝对路径当成服务器可用路径。到服务器后使用 prepare_data.py 重新生成索引，仍验证同一个冻结 manifest。
- 模型权重优先在服务器直接从官方源下载并核对 revision/hash，避免先占满本机再重复上传；是否能联网先测试。
- Windows PowerShell bootstrap.ps1 不在 Linux 上运行。先检查镜像内已有 PyTorch 与 CUDA，确认兼容后再在独立环境补依赖；不照搬本机 CUDA 安装命令。

## 6. 服务器上的首轮检查命令

将迁移包内容放到 code/，将原始训练包放到 datasets/rs_grounding_train_v1 后：

```bash
cd "$HOME/rs_grounding_z"
python code/z_pipeline/hpc/probe_runtime.py --out runs/runtime_check.json
python -m unittest discover -s code/z_evaluation -p 'test_*.py'
python -m unittest discover -s code/z_pipeline -p 'test_*.py'
python code/z_pipeline/prepare_data.py \
  --data datasets/rs_grounding_train_v1 \
  --out generated/data_v2
python code/z_evaluation/evalkit.py build \
  --data datasets/rs_grounding_train_v1 \
  --out generated/eval_v2
```

probe_runtime.py 不安装软件、不下载模型、不启动训练；仅做环境探测与极小 CUDA 运算。没有 PyTorch/CUDA 时会明确记录未通过。已有非空 generated/data_v2 时，导出脚本拒绝覆盖，复跑请使用新输出目录。

环境探测和数据检查通过后，才进入真实模型基线。baseline.py 和 diagnose.py 尚未完成，不能把计划中的命令当成现成训练程序。

## 7. 训练运行与结束

GPU 案例建议 nohup 加日志重定向，以便页面关闭后继续运行。等具体训练入口完成并验证后，再用类似下列方式运行（train_entry.py 为占位示例，不可直接执行）：

```bash
nohup python -u train_entry.py > runs/train.log 2>&1 &
tail -f runs/train.log
```

每次实验保存参数、随机种子、训练划分 hash、权重来源、检查点、dev 指标与耗时。补充 51 张验证图只用于阶段检查，不进入训练。

GPU 常见问题手册说明：实例进入“运行中”开始计时计费；关闭 Terminal 不等于释放；销毁实例才结束该实例计时。任务完成后先确认所有结果都在持久化目录并另存关键检查点，再按账户实际权限释放本人创建且不再使用的实例。共享或导师提供的实例应按分配约定处理，不能擅自销毁。

不支持 apt/apt-get 的环境应使用已提供镜像、用户目录或独立 Conda 环境；不改系统驱动，也不删除共享 Conda 环境。

## 8. 本项目阶段安排的变化

| 阶段 | 本机职责 | GPU 平台职责 |
|---|---|---|
| P0 基础验证 | 保存原始包、维护冻结划分、轻量测试 | 验证实际环境、重建 Linux 数据索引与评测 |
| P1 真实基线 | 查看配置和结果 | 下载原始模型、运行预测、评分与错例诊断 |
| P2 检测微调 | 审核对照设计 | D-FINE-S 微调、整图与切片实验 |
| P3 图文融合 | 分析类别与图像级结果 | 图文模型筛选与适配、关系模块消融 |
| P4 蒸馏 | 决定是否保留有益模块 | 教师推理、学生训练、同预算对照 |
| P5 验收 | 下载关键结果、维护 Z 分支 | 在支持容器构建的环境中做离线与计时检查；最终仍按比赛平台验收 |

远程算力能扩大实验空间，不改变评测标准，也不自动保证模型效果。

## 9. 已读取的资料

1. 自强5000快速使用手册.pdf：账号激活、校内登录、Slurm 与禁止登录节点计算。
2. 上海大学自强5000超级计算平台GPU计算资源实例介绍.pdf：A10040GB 配置与共享存储。
3. 上海大学自强5000超级计算平台GPU计算资源案例说明(pytorch镜像).pdf：教师授权、Terminal、上传、Conda、nohup 和 GPU 状态。
4. 上海大学自强5000超级计算平台GPU计算资源使用常见问题.pdf：计时计费、持久化目录、实例销毁、共享环境。
5. 队列情况说明.pdf：CPU 队列与机时约束。
6. 服务端口转发.pdf：需要查看容器内可视化服务时的校内端口转发；当前训练不要求开放服务端口。
7. 当前公开平台入口：https://hpc5000.shu.edu.cn/ 。硬件与界面会变化，账号实际显示优先。
