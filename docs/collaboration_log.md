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
- 姓名：Codex
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


