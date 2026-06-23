# 机器学习课程项目流程与三人分工方案

项目名称：多模态用户交互意图识别  
仓库名称：`multimodal-intention-recognition`  
项目目标：在老师提供的多模态意图识别代码基础上，完成端到端重构，并针对模态噪声和模态缺失问题进行实验与改进。

---

## 1. 项目要求理解

本课程项目的核心要求可以概括为：

1. 数据划分：
   - 用户 A、用户 B：训练集。
   - 用户 C：测试集。

2. 代码重构：
   - 原始代码流程是先提取特征、保存特征文件，再读取特征文件进行多模态融合训练。
   - 本项目需要将训练和测试流程改造成端到端流程，即：
     `raw data -> 数据预处理/特征处理 -> 多模态融合模型 -> 意图类别输出`。
   - 需要在报告中放置代码运行截图，证明端到端代码可以跑通。

3. 模态噪声实验：
   - 分别对每个单模态原始数据加入 20%、40%、60% 噪声。
   - 每种噪声设置下训练模型，并在用户 C 测试集上测试分类精度。

4. 模态缺失实验：
   - 分别丢弃任意一个模态、任意两个模态。
   - 每种缺失设置下训练模型，并在用户 C 测试集上测试分类精度。

5. 模型改进：
   - 在重构后的代码基础上，引入新模块或损失项。
   - 目标是提高模态噪声和模态缺失条件下的分类准确率。

6. 报告与答辩：
   - 报告不少于 15 篇参考文献。
   - 报告中需要写明方法改进、创新点、各成员任务和贡献比例。
   - 小组答辩为 5 分钟展示 + 2 分钟问答。

---

## 2. 当前仓库状态判断

当前仓库已经具备项目基础框架，主要内容如下：

```text
multimodal-intention-recognition/
├── README.md
├── README_CHINESE.md
├── requirements.txt
├── 课程项目2026.pdf
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── models/
│   │   └── baseline_real_scene.py
│   └── modules/
│       ├── real_scene_utils.py
│       └── feature_extraction/
├── experiments/
│   └── train_and_test.py
├── docs/
│   └── original_code_readme.md
├── results/
├── figures/
└── report/
```

当前仓库适合作为正式项目仓库继续开发，但还需要重点完成以下工作：

- 将 `experiments/train_and_test.py` 拆分或重构为更清晰的训练、测试、实验入口。
- 将 `src/modules/feature_extraction/` 中的特征提取脚本整合进训练和测试流程。
- 将硬编码路径改为项目相对路径或配置文件路径。
- 增加噪声实验、缺失模态实验和改进模型实验的独立运行入口。
- 统一保存指标、日志、预测结果、模型权重和实验图片。
- 增加团队协作日志，方便文稿负责人后续写报告。

---

## 3. 建议修改和新增的文件

### 3.1 需要重点修改的已有文件

| 文件 | 修改目标 |
|---|---|
| `experiments/train_and_test.py` | 作为原始参考代码保留，逐步拆分训练、测试和实验逻辑 |
| `src/models/baseline_real_scene.py` | 保留基础模型结构，抽取可复用模型类，避免训练逻辑和模型定义混在一起 |
| `src/modules/real_scene_utils.py` | 检查路径、视频映射、场景特征读取和缓存逻辑 |
| `src/modules/feature_extraction/*.py` | 将各模态特征提取封装成可被训练/测试脚本调用的函数 |
| `README.md` / `README_CHINESE.md` | 随项目进展补充运行方式、实验结果和最终目录说明 |
| `requirements.txt` | 根据实际运行报错补充缺失库，避免加入不使用的库 |

### 3.2 建议新增的文件

```text
configs/
├── default.yaml                  # 数据路径、训练参数、输出路径
├── noise.yaml                    # 噪声实验配置
└── missing_modality.yaml          # 缺失模态实验配置

src/
├── data/
│   ├── dataset.py                # 多模态数据集类
│   ├── build_samples.py          # 从 raw data 构建样本索引
│   └── transforms.py             # 噪声、缺失、归一化等处理
├── training/
│   ├── train_utils.py            # 训练循环、保存模型、日志记录
│   └── evaluate.py               # 准确率、F1、混淆矩阵等评估
└── utils/
    ├── paths.py                  # 项目路径管理
    ├── logger.py                 # 日志工具
    └── seed.py                   # 随机种子设置

experiments/
├── train.py                      # 干净数据训练入口
├── test.py                       # 测试入口
├── run_clean_baseline.py          # 清洁数据 baseline
├── run_noise_baseline.py          # 模态噪声 baseline
├── run_missing_baseline.py        # 模态缺失 baseline
└── run_improved_model.py          # 改进模型实验

docs/
├── experiment_log.md             # 每次实验记录
├── collaboration_log.md          # 团队协作日志
├── method_notes.md               # 方法和创新点记录
└── report_outline.md             # 报告大纲
```

---

## 4. 项目完成流程规划

### 阶段 0：环境和数据确认

目标：确认代码能运行，数据路径正确，依赖安装完整。

需要完成：

1. 创建环境并安装依赖：

```bash
pip install -r requirements.txt
```

2. 检查本地大文件是否存在：

```text
data/raw/imu.csv
data/raw/user_A/
data/raw/user_B/
data/raw/user_C/
data/raw/models/all-MiniLM-L6-v2/
data/raw/models/clip_teacher_model/
```

3. 检查 GPU 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

4. 运行最小化代码检查：

```bash
python experiments/train_and_test.py
```

如果原始脚本不能直接运行，先记录报错，不要立即大改代码。优先判断是否是路径、依赖、数据缺失或导入问题。

---

### 阶段 1：整理端到端数据流

目标：明确每个模态从 raw data 到模型输入特征的完整流程。

需要完成：

1. 梳理各模态来源：

| 模态 | 原始数据 | 处理脚本 | 模型输入 |
|---|---|---|---|
| scene | fisheye `.avi` | `real_scene_utils.py` | 场景视觉特征 |
| gesture | fisheye `.avi` | `strong_gesture2.0.py` | 手势特征 |
| audio | HoloLens `.mp4` | `mfcc.py` | 音频特征 |
| text | HoloLens `.mp4` | `ASR.py` | 语音文本特征 |
| imu | `imu.csv` | `imu.py` / `get_timestamp.py` | IMU 时序特征 |

2. 建立样本索引：
   - 每条交互视频对应一个样本。
   - 每个样本需要包含：用户、场景、意图标签、视频路径、IMU 时间段、各模态特征。

3. 输出建议：

```text
results/logs/sample_build_log.txt
results/metrics/sample_statistics.csv
```

---

### 阶段 2：代码重构为端到端训练和测试

目标：让训练和测试脚本可以从 raw data 开始执行，而不是手动读取已经保存好的特征文件。

建议拆分：

```text
experiments/train.py
experiments/test.py
```

`train.py` 需要完成：

```text
读取配置 -> 构建训练样本 -> 提取/缓存特征 -> 构建 DataLoader -> 训练模型 -> 保存模型与日志
```

`test.py` 需要完成：

```text
读取配置 -> 构建测试样本 -> 提取/缓存特征 -> 加载模型 -> 输出预测结果与评价指标
```

建议运行命令：

```bash
python experiments/train.py --config configs/default.yaml
python experiments/test.py --config configs/default.yaml --checkpoint results/checkpoints/best.pt
```

需要保存：

```text
results/checkpoints/best.pt
results/checkpoints/final.pt
results/checkpoints/scalers.pkl
results/checkpoints/label_encoder.pkl
results/metrics/clean_baseline_metrics.csv
results/logs/train_clean.log
results/predictions/clean_baseline_predictions.csv
figures/curves/clean_training_curve.png
figures/result_charts/clean_confusion_matrix.png
```

---

### 阶段 3：清洁数据 baseline 实验

目标：获得没有噪声、没有缺失模态时的基础模型表现。

需要运行：

```bash
python experiments/run_clean_baseline.py --config configs/default.yaml
```

需要记录：

| 指标 | 说明 |
|---|---|
| Accuracy | 主分类准确率 |
| Macro F1 | 多类别平均表现 |
| Weighted F1 | 考虑样本数后的 F1 |
| Confusion Matrix | 分析易混淆类别 |
| Training Time | 总训练时间 |
| Avg Train Time / Sample | 平均单样本训练时间 |
| Avg Test Time / Sample | 平均单样本测试时间 |

---

### 阶段 4：模态噪声 baseline 实验

目标：测试模型在单模态噪声干扰下的鲁棒性。

需要对每个模态分别加噪：

```text
imu, gesture, audio, text, scene
```

噪声比例：

```text
20%, 40%, 60%
```

建议运行：

```bash
python experiments/run_noise_baseline.py --config configs/noise.yaml
```

建议输出表格：

```text
results/metrics/noise_baseline_metrics.csv
figures/result_charts/noise_accuracy_comparison.png
figures/result_charts/noise_f1_comparison.png
```

报告中建议展示：

| Noise Modality | Noise Ratio | Accuracy | Macro F1 |
|---|---:|---:|---:|
| imu | 20% | 待填 | 待填 |
| imu | 40% | 待填 | 待填 |
| imu | 60% | 待填 | 待填 |
| gesture | 20% | 待填 | 待填 |
| ... | ... | ... | ... |

---

### 阶段 5：模态缺失 baseline 实验

目标：测试模型在单模态或双模态缺失条件下的分类能力。

单模态缺失：

```text
-drop imu
-drop gesture
-drop audio
-drop text
-drop scene
```

双模态缺失：

```text
-drop imu+gesture
-drop imu+audio
-drop imu+text
-drop imu+scene
-drop gesture+audio
-drop gesture+text
-drop gesture+scene
-drop audio+text
-drop audio+scene
-drop text+scene
```

建议运行：

```bash
python experiments/run_missing_baseline.py --config configs/missing_modality.yaml
```

建议输出：

```text
results/metrics/missing_modality_metrics.csv
figures/result_charts/missing_modality_accuracy_comparison.png
```

---

### 阶段 6：模型优化与创新实验

目标：选择一个主要创新方向深入实现，其他方向可以作为备选或报告展望。

建议至少完成一个“能稳定跑通、能对比 baseline”的创新点，不建议同时尝试过多复杂方法。

#### 方向 A：模态可靠性门控融合

思路：

- 为每个模态增加一个 reliability gate。
- 模型根据模态特征自动学习该模态是否可信。
- 噪声较大或缺失时降低该模态权重。

优点：

- 与模态噪声和缺失问题高度相关。
- 容易在报告中解释。
- 适合在现有多模态融合模型上修改。

适合作为首选创新方向。

---

#### 方向 B：模态 Dropout 训练

思路：

- 训练时随机屏蔽一个或多个模态。
- 让模型在训练阶段提前适应模态缺失。
- 测试时在缺失模态条件下应更稳定。

优点：

- 实现简单。
- 对缺失模态实验直接有效。
- 可以与方向 A 结合。

适合作为保底创新方向。

---

#### 方向 C：噪声一致性损失

思路：

- 同一个样本生成干净版本和加噪版本。
- 要求两者预测分布保持一致。
- 在交叉熵损失外增加 consistency loss。

示例损失：

```text
Loss = CrossEntropyLoss(clean) + CrossEntropyLoss(noisy) + λ * KL(pred_clean, pred_noisy)
```

优点：

- 针对噪声鲁棒性。
- 报告中容易作为“新损失项”说明。

缺点：

- 训练时间会增加。
- 需要调节 λ。

---

#### 方向 D：跨模态补全模块

思路：

- 当某个模态缺失时，用其他模态预测或重建该模态的隐层特征。
- 再将补全后的特征输入融合模块。

优点：

- 创新性较强。

缺点：

- 实现难度较高。
- 数据量较小时可能不稳定。

适合作为高风险高收益方向，不建议作为唯一方案。

---

#### 方向 E：分层融合结构

思路：

- 先融合强相关模态，例如 gesture + scene、audio + text。
- 再将视觉组、语音组、IMU 进行高层融合。

优点：

- 结构清晰，符合多模态任务逻辑。
- 可视化和报告表达比较方便。

缺点：

- 需要改动模型结构。
- 需要更多消融实验支撑。

---

## 5. 建议采用的最终技术路线

为了保证项目能够按时完成，建议采用以下组合：

```text
端到端重构
+ 清洁 baseline
+ 噪声 baseline
+ 缺失 baseline
+ 模态 Dropout
+ 模态可靠性门控融合
```

原因：

- 与课程要求完全对应。
- 实现难度可控。
- 创新点容易解释。
- 实验结果可以形成完整对比。

最终实验对比建议：

| 实验组 | 目的 |
|---|---|
| Clean Baseline | 正常条件下基础性能 |
| Noise Baseline | 加噪条件下基础模型鲁棒性 |
| Missing Baseline | 缺失模态条件下基础模型鲁棒性 |
| Improved Model | 验证改进模块是否有效 |
| Ablation Study | 验证每个创新模块的贡献 |

---

## 6. 三人合作分工方案

假设小组三人分别为：

- 成员 A：代码负责人 1，主要负责数据流和端到端重构。
- 成员 B：代码负责人 2，主要负责实验、模型改进和结果分析。
- 成员 C：文稿负责人，只负责报告、PPT、日志整理和答辩材料。

### 6.1 成员 A：端到端流程与数据工程

主要任务：

1. 检查仓库结构和数据路径。
2. 处理 `data/raw/` 下用户 A、B、C 的数据读取。
3. 整合各模态特征提取脚本。
4. 编写或维护：
   - `src/data/dataset.py`
   - `src/data/build_samples.py`
   - `src/modules/feature_extraction/` 中相关接口
   - `experiments/train.py`
   - `experiments/test.py`
5. 保证端到端流程可以跑通。
6. 保存运行截图和基础日志。

交付物：

```text
可运行的 train.py / test.py
样本构建日志
端到端运行截图
数据流说明文档
```

建议贡献比例：35%。

---

### 6.2 成员 B：实验设计、模型改进与结果分析

主要任务：

1. 运行清洁数据 baseline。
2. 实现和运行模态噪声实验。
3. 实现和运行模态缺失实验。
4. 实现至少一个模型改进方向。
5. 进行实验结果对比和消融实验。
6. 生成图表和指标文件。

重点维护文件：

```text
experiments/run_clean_baseline.py
experiments/run_noise_baseline.py
experiments/run_missing_baseline.py
experiments/run_improved_model.py
src/data/transforms.py
src/models/improved_model.py
src/training/evaluate.py
```

交付物：

```text
results/metrics/*.csv
results/logs/*.log
results/predictions/*.csv
figures/curves/*.png
figures/result_charts/*.png
模型改进说明
实验结论草稿
```

建议贡献比例：35%。

---

### 6.3 成员 C：文稿、PPT 与团队协作记录

成员 C 只负责文稿编写，不直接承担核心代码开发。

主要任务：

1. 维护项目日志：
   - `docs/experiment_log.md`
   - `docs/collaboration_log.md`
   - `docs/method_notes.md`
2. 整理课程项目报告。
3. 整理不少于 15 篇参考文献。
4. 根据成员 A、B 的实验结果编写：
   - 项目背景
   - 方法说明
   - 代码结构说明
   - 实验设置
   - 结果分析
   - 创新点说明
   - 成员贡献比例
5. 制作答辩 PPT。
6. 整理答辩讲稿和可能问答。

交付物：

```text
report/project_report.docx 或 .pdf
report/references.md
report/screenshots/
答辩 PPT
答辩讲稿
Q&A 准备文档
```

建议贡献比例：30%。

注意：

- 文稿负责人不应凭空编造实验结果。
- 所有报告结果必须来自 `results/metrics/`、`results/logs/` 和 `figures/`。
- 代码负责人每次完成实验后需要同步更新日志，方便文稿负责人写报告。

---

## 7. 团队协作流程建议

### 7.1 分支建议

```text
main                         # 稳定版本
feature/e2e-refactor          # 成员 A 使用
feature/experiments           # 成员 B 使用
feature/report-writing        # 成员 C 使用
```

### 7.2 每次提交建议格式

```text
[type] 简短说明
```

示例：

```text
[refactor] split train and test pipeline
[feat] add missing modality experiment runner
[exp] add noise baseline results
[docs] update experiment log and report outline
[fix] fix project-relative data path
```

### 7.3 每次完成代码修改后必须同步的内容

代码负责人需要同步：

```text
1. 修改了哪些文件
2. 为什么修改
3. 如何运行
4. 输出文件保存在哪里
5. 当前结果如何
6. 是否存在问题
```

建议写入：

```text
docs/collaboration_log.md
docs/experiment_log.md
```

---

## 8. 报告建议结构

```text
1. Introduction
2. Related Work
3. Dataset and Task Definition
4. Baseline Method
5. End-to-End Refactoring
6. Modal Noise Baseline
7. Missing Modality Baseline
8. Proposed Improvement
9. Experiments and Results
10. Ablation Study
11. Analysis and Discussion
12. Conclusion
13. Team Contribution
14. References
```

报告中必须包含：

- 数据集划分说明。
- 端到端代码运行截图。
- 模型结构图。
- 噪声实验表格。
- 缺失模态实验表格。
- 改进模型对比表格。
- 准确率和 F1 曲线或柱状图。
- 混淆矩阵。
- 每个成员任务和贡献比例。
- 不少于 15 篇参考文献。

---

## 9. 推荐优先级

如果时间有限，建议按照以下优先级完成：

1. 跑通原始代码。
2. 跑通端到端 `train.py` 和 `test.py`。
3. 得到 clean baseline 结果。
4. 得到模态噪声实验结果。
5. 得到模态缺失实验结果。
6. 实现模态 Dropout。
7. 实现模态可靠性门控融合。
8. 完成结果图表和报告。
9. 准备答辩 PPT。

---

## 10. 风险与应对

| 风险 | 应对方式 |
|---|---|
| 原始代码路径混乱 | 统一使用 `Path(__file__).resolve()` 和项目根目录路径 |
| 大文件无法上传 GitHub | 数据集和模型权重本地保存，仓库只保留说明和小型结果文件 |
| 特征提取耗时过长 | 增加缓存机制，缓存放入 `data/processed/` |
| 实验组合太多 | 优先做完整 baseline，再选择少量关键组合深入分析 |
| 改进模型效果不稳定 | 保留 baseline 结果，报告中如实分析原因 |
| 文稿负责人缺少材料 | 每次实验后强制更新日志和图表路径 |

---

## 11. 最终提交检查清单

提交前检查：

- [ ] `train.py` 可以从 raw data 运行。
- [ ] `test.py` 可以加载模型并输出预测结果。
- [ ] 用户 A、B 作为训练集，用户 C 作为测试集。
- [ ] clean baseline 有结果。
- [ ] noise baseline 有结果。
- [ ] missing modality baseline 有结果。
- [ ] improved model 有对比结果。
- [ ] 结果保存在 `results/`。
- [ ] 图片保存在 `figures/`。
- [ ] 报告截图保存在 `report/screenshots/`。
- [ ] 报告写明创新点。
- [ ] 报告写明成员任务和贡献比例。
- [ ] 参考文献不少于 15 篇。
- [ ] README 已更新运行方式。
- [ ] 最终代码压缩包包含必要模型文件。
