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

## 2026-06-23 - 实验记录文档创建

### 贡献者
- 姓名：Mplus0
- 角色：Experiment / Report

### 修改文件
- `docs/experiment_log.md`：创建实验记录文档，包含统一实验设置、待执行实验矩阵和各类实验记录模板。
- `docs/collaboration_log.md`：追加本次实验记录文档创建记录。

### 修改目的
为成员 B 后续记录 clean baseline、模态噪声实验、模态缺失实验、改进模型实验和消融实验提供统一模板，方便成员 C 后续整理报告材料。

### 运行方式
本次任务只涉及文档，不需要运行命令。

### 输出文件
- `docs/experiment_log.md`
- `docs/collaboration_log.md`

### 当前状态
- 已完成

### 给报告撰写者的说明
当前 `docs/experiment_log.md` 只记录实验计划和模板，所有指标均为待填写。后续报告中的实验结果必须来自真实运行后的 `results/metrics/`、`results/logs/`、`results/predictions/` 和 `figures/`。

### 遗留问题
- 成员 A 的端到端训练和测试入口尚未完成。
- 当前未运行任何训练或测试实验。

## 2026-06-23 - Baseline 配置与实验日志模板准备

### 贡献者
- 姓名：Mplus0
- 角色：Experiment / Report

### 修改文件
- `configs/noise.yaml`：创建 modal noise baseline 配置，包含实验名称、随机种子、数据划分、模态列表、噪声比例、单模态加噪策略、baseline 模型类型和输出路径。
- `configs/missing_modality.yaml`：创建 missing-modality baseline 配置，包含实验名称、随机种子、数据划分、完整模态列表、单模态缺失组合、双模态缺失组合、baseline 模型类型和输出路径。
- `docs/experiment_log.md`：追加 Clean Baseline、Modal Noise Baseline 和 Missing Modality Baseline 的基础实验配置说明与记录模板。
- `docs/collaboration_log.md`：追加本次 baseline 配置准备记录。

### 修改目的
为成员 B 后续编写 `experiments/run_clean_baseline.py`、`experiments/run_noise_baseline.py` 和 `experiments/run_missing_baseline.py` 提供清晰的配置依据，并为成员 C 后续整理实验报告提供统一模板。

### 运行方式
本次任务仅准备配置文件和实验日志模板，暂不运行实验。

### 输出文件
- `configs/noise.yaml`
- `configs/missing_modality.yaml`
- `docs/experiment_log.md`
- `docs/collaboration_log.md`

### 当前状态
- 已完成配置准备

### 给报告撰写者的说明
当前没有运行任何实验，也没有真实指标结果。`docs/experiment_log.md` 中新增的基础实验字段均为 TBD，后续必须从真实运行后的 `results/metrics/`、`results/logs/`、`results/predictions/` 和 `figures/` 同步。

### 遗留问题
- 需要等待成员 A 的 `train.py` / `test.py` 接口稳定。
- 后续再编写 `experiments/run_clean_baseline.py`、`experiments/run_noise_baseline.py` 和 `experiments/run_missing_baseline.py`。
