## 2026-06-22 - 路径配置规范化

### 贡献者
- 姓名：Mplus0
- 角色：文档 / 配置

### 修改文件
- `configs/default.yaml`：创建项目相对路径的统一配置约定文件。
- `docs/path_setup.md`：创建中文路径设置说明，解释本地数据集、模型、缓存、输出、图表和报告材料的放置规则。

### 修改目的
标准化团队协作中的文件夹路径约定，便于成员理解本地数据、模型、缓存、中间文件和实验输出应放置的位置。本次修改不改变现有源码或实验流程。

### 运行方式
本次任务只涉及配置和文档，不需要运行命令。

### 输出文件
- `configs/default.yaml`
- `docs/path_setup.md`

### 当前状态
- 已完成

### 给报告撰写者的说明
本步骤只记录本地数据集、本地模型、缓存文件、处理后数据、实验结果、图表和报告材料的推荐放置位置。

### 遗留问题
- 当前训练和测试代码尚未接入 `configs/default.yaml`。
- 本次任务未运行任何实验。

## 2026-06-22 - 路径工具文件创建

### 贡献者
- 姓名：Mplus0
- 角色：代码 / 文档

### 修改文件
- `src/utils/paths.py`：创建轻量级路径工具文件，用于读取 `configs/default.yaml` 并解析项目相对路径。
- `docs/collaboration_log.md`：追加本次路径工具创建记录。

### 修改目的
为团队协作提供可复用的路径读取、路径解析、运行时目录创建、Hugging Face 缓存环境变量设置和路径报告打印工具。当前工具尚未接入 baseline、训练、测试或特征提取流程。

### 运行方式
```bash
python src/utils/paths.py
```

### 输出文件
- `src/utils/paths.py`
- `docs/collaboration_log.md`

### 当前状态
- 已完成

### 给报告撰写者的说明
本次修改只提供路径配置读取工具，不涉及实验结果、模型结构或训练流程变化。

### 遗留问题
- 运行 `src/utils/paths.py` 前需要确保 `configs/default.yaml` 已存在。
- 本次任务未运行任何实验。

## 2026-06-23 - 日志与随机种子工具补充

### 贡献者
- 姓名：Mplus0
- 角色：代码 / 文档

### 修改文件
- `src/utils/logger.py`：新增项目日志工具，支持控制台日志、文件日志、实验日志路径生成和重复配置时的 handler 清理。
- `src/utils/seed.py`：新增随机种子工具，统一设置 Python、NumPy 和 PyTorch 的随机种子，并提供 DataLoader worker 和 generator 辅助函数。
- `docs/collaboration_log.md`：追加本次工具补充记录。

### 修改目的
为后续训练、测试和实验脚本提供统一的日志记录与可复现实验设置，减少不同成员各自实现日志和随机种子逻辑造成的不一致。

### 运行方式
```bash
python src/utils/logger.py
python src/utils/seed.py
```

### 输出文件
- `src/utils/logger.py`
- `src/utils/seed.py`
- `docs/collaboration_log.md`

### 当前状态
- 已完成

### 给报告撰写者的说明
本次修改属于工程规范化工作，不涉及模型结构、训练流程或实验结果变化。后续实验脚本可以使用这些工具记录日志并固定随机种子。

### 遗留问题
- 当前训练、测试和特征提取脚本尚未接入 `src/utils/logger.py` 和 `src/utils/seed.py`。
- 本次任务未运行任何训练或测试实验。

## 2026-06-24 - 成员 B baseline 接口接入

### Contributor
- Name: Codex
- Role: Code / Experiment

### Files Changed
- `src/utils/paths.py`：补充 `load_config()` 兼容入口，并把 `outputs.checkpoint_dir` 纳入运行目录创建和路径检查。
- `src/data/transforms.py`：实现 `ModalNoiseTransform`、`MissingModalityTransform` 和 `ComposeTransforms`，保留 `apply_modal_noise()` 与 `apply_missing_modalities()` 函数式接口。
- `src/training/experiment_runner.py`：新增 clean、modal noise、missing modality baseline 可复用训练与测试 runner。
- `experiments/run_clean_baseline.py`：改为调用基础 runner 执行 clean baseline。
- `experiments/run_noise_baseline.py`：改为读取 `configs/noise.yaml` 的实验矩阵并循环执行单模态噪声 baseline。
- `experiments/run_missing_baseline.py`：改为读取 `configs/missing_modality.yaml` 的实验矩阵并循环执行单/双模态缺失 baseline。
- `docs/collaboration_log.md`：追加本次接口接入记录。

### Purpose
根据 `MEMBER_B_INTERFACE_INTEGRATION_PLAN.md` 的基础方案，将成员 B 预留的 `TODO` / placeholder 接口接入成员 A 已提供的 Dataset、baseline model 和 training engine。此次只完成 clean baseline、modal noise baseline 和 missing modality baseline 的基础代码接口，不实现 improved model、新模块或新损失项。

### How to Run
```bash
python experiments/run_clean_baseline.py --config configs/default.yaml
python experiments/run_noise_baseline.py --config configs/noise.yaml
python experiments/run_missing_baseline.py --config configs/missing_modality.yaml
```

### Output Files
- `results/metrics/clean_baseline_metrics.csv`
- `results/metrics/noise_baseline_metrics.csv`
- `results/metrics/missing_modality_metrics.csv`
- `results/logs/*.log`
- `results/predictions/*_predictions.csv`
- `results/checkpoints/*_best.pt`
- `results/checkpoints/*_final.pt`
- `figures/*_loss_curve.png`
- `figures/*_confusion_matrix.png`

### Current Status
- Partially completed

### Notes for Report Writer
本次修改只补齐实验运行接口和结果保存路径，不产生正式实验结果。正式结果必须在课程服务器准备好数据集、依赖和本地模型后运行上述命令生成，不应使用 smoke-test 结果写入报告。

### Remaining Problems
- 当前本机缺少课程数据集和依赖，因此未运行正式训练、测试或特征提取。
- 噪声 baseline 目前在 Dataset feature 层实现可复现实验扰动；若后续需要严格解释为 raw data 噪声，应在成员 A 提供 raw-level hook 后前移扰动位置。
- improved model、新模块和新损失项尚未实现，留待后续任务。

## 2026-06-25 - 模型优化任务方案整理

### Contributor
- Name: Codex
- Role: Report / Experiment Planning

### Files Changed
- `MODEL_OPTIMIZATION_TASK_PLAN.md`：新增模型优化任务详细方案，说明可靠性门控融合、模态 Dropout 训练、建议新增文件、实验矩阵、输出规范和报告表述。
- `docs/collaboration_log.md`：追加本次方案整理记录。

### Purpose
在不修改已验证 baseline 代码的前提下，为后续 improved model 任务提供可执行的设计方案。该方案优先复用当前 Dataset、特征缓存、噪声和缺失实验接口，并把模型优化重点放在 `modality_mask` 可感知的可靠性门控融合上。

### How to Run
本次任务只新增方案文档，不运行训练或测试命令。

### Output Files
- `MODEL_OPTIMIZATION_TASK_PLAN.md`
- `docs/collaboration_log.md`

### Current Status
- Completed; local syntax check passed

### Notes for Report Writer
后续报告中的“方法改进与创新点”可参考 `MODEL_OPTIMIZATION_TASK_PLAN.md` 中的“报告可复用描述草稿”。正式结果仍需在课程服务器完成 improved model 实现和全量实验后填写。

### Remaining Problems
- 本次只生成模型优化方案，尚未实现 `src/models/improved_model.py`、`configs/improved_model.yaml` 或 `experiments/run_improved_model.py`。
- 本机缺少数据集和依赖，未运行任何正式实验。

## 2026-06-25 - 第一阶段：新增可靠性门控模型

### Contributor
- Name: Codex
- Role: Code

### Files Changed
- `src/models/improved_model.py`：新增 `ReliabilityGatedMultimodalModel`、`build_improved_model_from_config()` 和合成数据 forward smoke-test 入口。
- `docs/collaboration_log.md`：追加第一阶段实现记录。

### Purpose
在不修改已验证 baseline 文件的前提下，新增 improved model 的模型定义。该模型沿用当前五模态 Dataset 输入约定，并在融合阶段加入 per-modality reliability gate 与 `modality_mask` 感知的加权池化，为后续噪声和缺失模态鲁棒实验提供模型基础。

### How to Run
```bash
python src/models/improved_model.py --config configs/default.yaml --smoke-test
```

### Output Files
- 本阶段不生成正式实验输出文件。

### Current Status
- Completed; local syntax check passed

### Notes for Report Writer
本阶段只完成模型结构定义，尚未训练模型，也没有正式指标。方法描述可概括为：对五个模态分别估计可靠性权重，并结合 `modality_mask` 抑制缺失模态对最终融合表示的影响。

### Remaining Problems
- 本机运行 `python src/models/improved_model.py --config configs/default.yaml --smoke-test` 时缺少 `torch`，未安装依赖；需要在课程服务器环境执行该 smoke test。
- 尚未新增 `RandomModalityDropoutTransform`。
- 尚未新增 `configs/improved_model.yaml`。
- 尚未新增 `experiments/run_improved_model.py` 或 improved runner。
- 需要在服务器环境运行 smoke test 和后续训练验证。

## 2026-06-25 - 第二阶段：新增模态 Dropout 训练增强

### Contributor
- Name: Codex
- Role: Code

### Files Changed
- `src/data/improved_transforms.py`：新增 `RandomModalityDropoutTransform`、`build_random_modality_dropout_from_config()` 和合成 sample smoke-test。
- `docs/collaboration_log.md`：追加第二阶段实现记录。

### Purpose
为 improved model 提供训练阶段的随机模态屏蔽增强，使模型在训练时接触单模态或双模态缺失组合。该 transform 与现有 `MissingModalityTransform` 使用一致的表示方式：被丢弃模态置零，并将对应 `modality_mask` 设为 `False`。

### How to Run
```bash
python src/data/improved_transforms.py
```

### Output Files
- 本阶段不生成正式实验输出文件。

### Current Status
- Completed

### Notes for Report Writer
本阶段只实现训练增强策略，尚未接入 improved runner，也没有正式指标。报告中可描述为：训练阶段随机屏蔽一个或两个模态，使模型提前适应课程要求中的模态缺失设置。

### Remaining Problems
- 本机运行 `python src/data/improved_transforms.py` 时缺少 `torch`，未安装依赖；需要在课程服务器环境执行该 smoke test。
- 尚未新增 `configs/improved_model.yaml`。
- 尚未新增 `experiments/run_improved_model.py` 或 improved runner。
- 尚未将 `RandomModalityDropoutTransform` 接入训练流程。

## 2026-06-25 - 第三阶段：新增 improved model 配置

### Contributor
- Name: Codex
- Role: Code / Experiment

### Files Changed
- `configs/improved_model.yaml`：新增 improved model 独立配置，包含可靠性门控参数、模态 Dropout 参数、clean/noise/missing/ablation 实验矩阵和输出前缀。
- `docs/collaboration_log.md`：追加第三阶段实现记录。

### Purpose
为后续 improved runner 提供独立配置文件，避免修改 `configs/default.yaml` 和现有 baseline 配置。该配置只保存 improved model 相关参数，正式数据路径、模态维度、输出目录和用户划分仍从 `configs/default.yaml` 读取。

### How to Run
本阶段只新增配置文件，不运行训练或测试命令。可用以下命令在服务器或本地检查 YAML 是否可解析：
```bash
python -c "import yaml; yaml.safe_load(open('configs/improved_model.yaml', encoding='utf-8')); print('improved config ok')"
```

### Output Files
- 本阶段不生成正式实验输出文件。

### Current Status
- Completed

### Notes for Report Writer
本阶段只提供 improved model 的实验配置，尚无正式指标。后续 clean、noise、missing 和 ablation 实验会从该配置读取模型和实验矩阵。

### Remaining Problems
- 本机运行配置解析命令时缺少 `yaml`，未安装依赖；需要在课程服务器环境执行配置解析检查。
- 尚未新增 `experiments/run_improved_model.py` 或 improved runner。
- 尚未将 improved model 与 `RandomModalityDropoutTransform` 接入训练流程。
- 需要在服务器环境运行配置解析和后续训练验证。

## 2026-06-25 - 第四阶段：新增 improved runner 和实验入口

### Contributor
- Name: Codex
- Role: Code / Experiment

### Files Changed
- `src/training/improved_experiment_runner.py`：新增 improved model 可复用实验 runner，复用现有 Dataset、engine、输出路径和 metrics table 工具，支持 clean、noise、missing 和 ablation。
- `experiments/run_improved_model.py`：新增 improved model 命令行入口，支持 `--mode all/clean/noise/missing/ablation`、`--epochs`、`--max-samples` 和 `--smoke-test`。
- `docs/collaboration_log.md`：追加第四阶段实现记录。

### Purpose
将前三阶段新增的可靠性门控模型、模态 Dropout transform 和 improved 配置接入独立训练流程，同时避免修改已验证的 baseline runner 和 clean/noise/missing baseline 入口。

### How to Run
短跑检查：
```bash
python experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode clean --smoke-test --epochs 1
```

正式运行：
```bash
python experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml
```

分阶段运行：
```bash
python experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode clean
python experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode noise
python experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode missing
python experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode ablation
```

### Output Files
- `results/metrics/improved_model_metrics.csv`
- `results/metrics/improved_clean_metrics.csv`
- `results/metrics/improved_noise_metrics.csv`
- `results/metrics/improved_missing_modality_metrics.csv`
- `results/metrics/improved_ablation_metrics.csv`
- `results/logs/improved_*.log`
- `results/predictions/improved_*_predictions.csv`
- `results/checkpoints/improved_*_best.pt`
- `results/checkpoints/improved_*_final.pt`
- `figures/improved_*_loss_curve.png`
- `figures/improved_*_confusion_matrix.png`

### Current Status
- Completed; local syntax check passed

### Notes for Report Writer
本阶段完成 improved model 的独立实验入口，但本机没有依赖和数据集，尚未产生正式结果。正式报告中的 improved 指标必须来自课程服务器完整运行后的 `results/metrics/improved_*.csv`。

### Remaining Problems
- 本机运行 improved runner smoke test 时缺少 `torch`，未安装依赖；需要在课程服务器环境执行。
- 需要上传服务器执行 `--smoke-test --epochs 1 --mode clean` 验证入口。
- 需要服务器全量运行 clean、noise、missing 和 ablation 后才能填写报告指标。

## 2026-06-25 - 第五阶段：本地轻量检查与服务器验证准备

### Contributor
- Name: Codex
- Role: Code / Experiment

### Files Changed
- `docs/collaboration_log.md`：追加第五阶段检查记录。

### Purpose
在不安装本机依赖、不运行真实数据训练的前提下，对前四阶段新增文件进行语法级检查，并整理需要上传课程服务器执行的验证命令。

### How to Run
本机已完成语法检查：
```bash
python -m py_compile src/models/improved_model.py src/data/improved_transforms.py src/training/improved_experiment_runner.py experiments/run_improved_model.py
```

服务器建议先运行以下 smoke test：
```bash
python src/models/improved_model.py --config configs/default.yaml --smoke-test
python src/data/improved_transforms.py
python experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode clean --smoke-test --epochs 1
```

服务器 smoke test 通过后，再运行真实数据短跑：
```bash
python experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode clean --max-samples 6 --epochs 1
```

### Output Files
- 本阶段不生成正式实验输出文件。
- 服务器 smoke test 若成功，会生成 `results/metrics/improved_model_metrics.csv` 以及对应 logs、predictions、checkpoints 和 figures。

### Current Status
- Completed; local syntax check passed

### Notes for Report Writer
本阶段仍不产生正式实验结果。所有 `--smoke-test` 和 `--max-samples` 输出只能作为代码检查证据，不能作为报告中的正式 Accuracy 或 F1 指标。

### Remaining Problems
- 本机缺少 `torch`、`yaml` 等依赖，无法执行运行级 smoke test。
- 需要上传课程服务器执行上述 smoke test 和真实数据短跑。
- 服务器验证通过后，才能进行 clean、noise、missing 和 ablation 的全量 improved 实验。

## 2026-06-25 - 服务器验证反馈修复：improved_transforms 直接运行导入路径

### Contributor
- Name: Codex
- Role: Code

### Files Changed
- `src/data/improved_transforms.py`：补充直接运行脚本时的项目根目录 `sys.path` 注入逻辑，使 `python src/data/improved_transforms.py` 能正确导入 `src.data.transforms`。
- `docs/collaboration_log.md`：追加本次服务器反馈修复记录。

### Purpose
服务器运行 `python src/data/improved_transforms.py` 时出现 `ModuleNotFoundError: No module named 'src'`。原因是该文件作为脚本直接执行时，Python 默认搜索路径不包含项目根目录。本次修复与项目中其他可直接运行文件的处理方式保持一致。

### How to Run
```bash
python src/data/improved_transforms.py
```

### Output Files
- 本次修复不生成正式实验输出文件。

### Current Status
- Completed; waiting for server re-test

### Notes for Report Writer
本次属于工程入口修复，不涉及模型方法或实验结果变化。

### Remaining Problems
- 需要将修复后的 `src/data/improved_transforms.py` 上传服务器后重新执行 smoke test。
- 其他 improved runner 命令仍需要服务器继续验证。

## 2026-06-25 - 服务器验证反馈修复：模态 Dropout 批处理字段

### Contributor
- Name: Codex
- Role: Code

### Files Changed
- `src/data/improved_transforms.py`：将 `dropped_modalities` 从变长 list 改为字符串，避免 PyTorch DataLoader 默认 collate 在 batch 内样本丢弃模态数量不一致时报错。
- `docs/collaboration_log.md`：追加本次服务器反馈修复记录。

### Purpose
服务器运行 improved clean smoke test 时，DataLoader 报错 `RuntimeError: each element in list of batch should be of equal size`。原因是 transform 给每个 sample 添加了长度不固定的 `dropped_modalities` list。本次修复保留该调试字段，但改为 `"none"` 或 `"imu+text"` 这类字符串，避免影响 batch collate。

### How to Run
```bash
python experiments/run_improved_model.py --config configs/improved_model.yaml --base-config configs/default.yaml --mode clean --smoke-test --epochs 1
```

### Output Files
- 本次修复不生成正式实验输出文件。

### Current Status
- Completed; waiting for server re-test

### Notes for Report Writer
本次属于训练数据 batch 组织修复，不涉及模型方法或正式实验结果变化。

### Remaining Problems
- 需要将修复后的 `src/data/improved_transforms.py` 上传服务器后重新执行 improved clean smoke test。
- 服务器通过 smoke test 后，再进行真实数据短跑。

## 2026-06-25 - 更新全项目运行与截图保存文档

### Contributor
- Name: Codex
- Role: Report / Experiment

### Files Changed
- `MEMBER_A_FORMAL_RUN_PLAN.md`：重写为全项目服务器运行手册，覆盖环境检查、数据与模型路径检查、样本索引、端到端特征构建、clean baseline、noise baseline、missing baseline、improved model、输出检查、终端日志保存和截图清单。
- `docs/collaboration_log.md`：追加本次运行文档更新记录。

### Purpose
将原先只覆盖成员 A clean 主线的运行方案扩展为完整课程项目全流程运行文档，方便服务器正式运行、报告截图收集、终端日志保存和最终结果核查。

### How to Run
本次任务只更新 Markdown 文档，不运行训练或测试命令。正式命令详见：
```bash
less MEMBER_A_FORMAL_RUN_PLAN.md
```

### Output Files
- `MEMBER_A_FORMAL_RUN_PLAN.md`
- `docs/collaboration_log.md`

### Current Status
- Completed

### Notes for Report Writer
报告撰写时可依据 `MEMBER_A_FORMAL_RUN_PLAN.md` 的截图清单检查证据是否齐全。正式实验指标仍必须来自完整数据运行后的 `results/metrics/*.csv`。

### Remaining Problems
- 需要按该文档在服务器上完成全量 clean、noise、missing 和 improved 实验。
- 需要将最终截图保存到 `report/screenshots/`，终端文本保存到 `report/terminal_logs/`。

## 2026-06-25 - 同步更新中英文 README

### Contributor
- Name: Codex
- Role: Report / Documentation

### Files Changed
- `README.md`：重写英文 README，整理当前仓库结构、已实现代码能力、运行入口、输出目录、论文编写材料和当前无正式结果文件的状态说明。
- `README_CHINESE.md`：同步重写中文 README，与英文版保持内容一致。
- `docs/collaboration_log.md`：追加本次 README 更新记录。

### Purpose
根据当前项目状态更新仓库首页说明，方便团队成员和报告撰写者快速理解仓库内容、运行入口、实验分组、输出位置和当前“代码已可运行但尚未正式训练、未放置结果文件”的状态。

### How to Run
本次任务只更新文档，不运行训练或测试命令。

### Output Files
- `README.md`
- `README_CHINESE.md`
- `docs/collaboration_log.md`

### Current Status
- Completed

### Notes for Report Writer
README 中已经整理了论文/报告编写可参考的文件列表、建议报告结构和正式指标来源说明。报告中的正式实验结果仍需来自服务器全量运行后的 `results/metrics/*.csv`。

### Remaining Problems
- 当前仓库仍未包含正式训练结果、checkpoint 或最终图表。
- 需要后续完成服务器全量训练，并将可提交的轻量结果表格或图表按课程要求整理。
