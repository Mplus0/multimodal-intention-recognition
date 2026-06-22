# Path Setup Guide

## 1. 文件作用

`configs/default.yaml` 是本项目的统一路径约定文件，用来说明团队成员应当把本地数据集、本地模型、缓存文件、中间处理数据、实验结果、图表和报告材料放在哪里。

当前文件只用于团队协作中的路径规范说明。它不改变现有训练、测试、特征提取或模型代码的运行方式。

## 2. 推荐目录结构

建议团队成员在仓库根目录下按照以下结构放置本地文件：

```text
data/
  raw/
    imu.csv
    user_A/
    user_B/
    user_C/
    models/
  processed/
    cache/
models/
results/
  metrics/
  logs/
  predictions/
figures/
report/
  screenshots/
```

以上路径均为项目相对路径。不同成员的本机目录位置可以不同，但进入仓库后应尽量保持相同的内部目录结构。

## 3. 数据集放置规则

`data/raw` 用于放置原始数据集文件，例如 IMU 表格、用户 A/B/C 的原始多模态数据，以及 HoloLens、fisheye 等原始子目录。

推荐示例：

```text
data/raw/imu.csv
data/raw/user_A/HoloLens/
data/raw/user_A/fisheye/
data/raw/user_B/HoloLens/
data/raw/user_C/fisheye/
```

`data/processed` 用于放置预处理后的数据、中间特征、样本索引或后续流程生成的可复用文件。原始数据和处理后数据应尽量分开，便于排查问题和复现实验。

## 4. 本地模型放置规则

`data/raw/models` 用于放置本地下载的预训练模型或教师代码依赖的模型文件。

推荐示例：

```text
data/raw/models/all-MiniLM-L6-v2/
data/raw/models/clip_teacher_model/
data/raw/models/clip_teacher_model/vit-base-patch16-224/
```

这些模型通常体积较大，不应上传到 GitHub。团队成员可以根据 `configs/default.yaml` 中的路径约定，在各自本地或服务器环境中准备相同目录。

## 5. 缓存与中间文件规则

`data/processed/cache` 用于放置运行过程中产生的缓存文件，包括 Hugging Face 缓存、场景特征缓存、已提取特征缓存等。

推荐示例：

```text
data/processed/cache/huggingface/hub/
data/processed/cache/huggingface/transformers/
data/processed/cache/scene_cache_real_vit/
data/processed/cache/feature_cache/
```

缓存文件可以帮助减少重复下载和重复计算，但它们不是源代码，也通常不适合提交到 GitHub。

## 6. 实验输出规则

`models` 用于保存训练得到的模型权重或检查点。

`results` 用于保存实验输出，建议继续细分为：

```text
results/metrics/
results/logs/
results/predictions/
```

`figures` 用于保存论文、报告或展示中需要使用的图表。

`report` 用于保存报告正文相关材料，`report/screenshots` 用于保存实验截图或运行证据。

## 7. 不应上传到 GitHub 的文件

以下内容通常不应上传到 GitHub：

- 原始数据集和用户采集数据。
- 本地预训练模型和下载后的模型权重。
- 训练得到的模型检查点。
- 大型缓存文件。
- 已提取的大型特征文件。
- 临时日志、重复生成的中间文件和本机环境文件。

如果需要共享这些文件，应通过课程服务器、网盘或其他团队约定的方式传递，并在文档中说明放置路径。

## 8. 路径工具 `paths.py` 使用说明

`src/utils/paths.py` 是一个轻量级路径工具文件，用于读取 `configs/default.yaml` 中的路径配置，并把项目相对路径解析为绝对路径。它的目标是帮助团队成员在不同机器或服务器环境中使用同一套路径约定。

该工具是独立文件，不会自动改变现有训练、测试、模型或特征提取代码的运行方式。

### 8.1 运行前要求

运行 `paths.py` 前需要确认：

- 仓库根目录下已经存在 `configs/default.yaml`。
- 当前 Python 环境已经安装 `PyYAML`，因为工具内部使用 `yaml.safe_load()` 读取 YAML 配置。
- 在课程服务器上，如果没有 `python` 命令，可以尝试使用 `python3`。

如果 `configs/default.yaml` 不存在，`paths.py` 会给出明确错误提示，要求先创建该配置文件。

### 8.2 直接检查路径配置

在仓库根目录运行：

```bash
python src/utils/paths.py
```

如果服务器环境只有 `python3`：

```bash
python3 src/utils/paths.py
```

直接运行时会执行以下操作：

- 读取 `configs/default.yaml`。
- 设置 Hugging Face 相关缓存环境变量，但不会覆盖已经存在的环境变量。
- 创建安全的运行时目录，例如 `data/processed`、`data/processed/cache`、`results/metrics`、`results/logs`、`figures`、`models`、`report/screenshots`。
- 打印路径检查报告，例如：

```text
[OK] data.raw_dir -> /abs/path/to/data/raw
[WARN] data.imu_csv -> /abs/path/to/data/raw/imu.csv
[OK] outputs.metrics_dir -> /abs/path/to/results/metrics
```

`[WARN]` 不一定代表程序错误。对于原始数据、预训练模型、缓存文件等本地大文件，缺失通常只表示当前环境尚未放置对应文件。

### 8.3 在其他 Python 文件中导入使用

如果后续脚本需要读取统一路径，可以按下面方式导入：

```python
from src.utils.paths import (
    load_paths_config,
    get_path,
    ensure_runtime_dirs,
    setup_huggingface_env,
    print_path_report,
)

config = load_paths_config()

raw_dir = get_path("data", "raw_dir", config=config)
imu_csv = get_path("data", "imu_csv", config=config)
metrics_dir = get_path("outputs", "metrics_dir", config=config)

setup_huggingface_env(config)
ensure_runtime_dirs(config)
print_path_report(config)

print(raw_dir)
print(imu_csv)
print(metrics_dir)
```

常用路径读取示例：

```python
get_path("data", "raw_dir", config=config)
get_path("data", "processed_dir", config=config)
get_path("local_models", "sentence_model", config=config)
get_path("cache", "huggingface", config=config)
get_path("outputs", "metrics_dir", config=config)
get_path("report", "screenshots_dir", config=config)
```

### 8.4 主要函数说明

- `get_project_root()`：自动检测仓库根目录。当前规则是从 `src/utils/paths.py` 向上两级定位项目根目录。
- `load_paths_config(config_path=None)`：读取 YAML 路径配置。默认读取 `configs/default.yaml`。
- `resolve_path(path_value)`：把相对路径转换为基于仓库根目录的绝对路径；如果传入值已经是绝对路径，则保持不变。
- `get_path(*keys, config=None)`：按嵌套 key 读取路径，例如 `get_path("data", "raw_dir")`。
- `ensure_dir(path)`：创建指定目录并返回该路径。
- `ensure_runtime_dirs(config=None)`：只创建安全的运行时输出目录，不自动创建原始数据目录。
- `setup_huggingface_env(config=None)`：根据配置设置 `HF_HOME`、`HF_HUB_CACHE`、`TRANSFORMERS_CACHE`，并使用 `os.environ.setdefault()` 避免覆盖已有环境变量。
- `validate_paths(config=None)`：检查重要路径是否存在，并返回字典形式的检查结果。
- `print_path_report(config=None)`：打印可读的路径检查报告。

### 8.5 注意事项

`paths.py` 目前只是团队协作辅助工具，尚未接入当前 baseline、训练、测试或特征提取流程。后续如果需要让现有代码正式使用这些路径函数，应另开任务，并在修改前确认影响范围。

不要把原始数据集、本地预训练模型、训练检查点、大型缓存或已提取的大型特征文件提交到 GitHub。

## 9. 当前阶段说明

当前阶段已经新增统一路径配置文件 `configs/default.yaml`、路径说明文档 `docs/path_setup.md`，以及独立路径工具 `src/utils/paths.py`。

当前阶段仍未修改训练、测试、模型、特征提取或数据处理代码。后续如需让现有代码读取 `configs/default.yaml` 或调用 `src/utils/paths.py`，应另开任务并在修改前确认影响范围。
