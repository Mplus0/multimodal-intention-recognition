下面是一份适合你当前阶段使用的 `README.md`，只介绍仓库背景和项目架构，不涉及具体代码文件实现，后续可以继续补充运行方式、实验结果和成员分工。

# Multimodal User Intention Recognition

## 1. Project Overview

This repository is created for the final project of the Machine Learning course.

The project focuses on **Multimodal User Intention Recognition** in augmented reality interaction scenarios. The main goal is to build and improve a multimodal learning framework that can recognize user intentions from raw multimodal interaction data.

This project will be completed by a team of three members. The final results will be used for course project submission and group defense. Therefore, this repository is organized with a clear and professional structure to support collaborative development, experiment management, report writing, and presentation preparation.

At the current stage, this repository mainly provides the overall project architecture. Specific source code files, experiment scripts, and implementation details will be added in later development stages.

---

## 2. Project Background

With the development of AR devices and intelligent interaction systems, traditional single-modal interaction methods are often insufficient for understanding complex user intentions. Multimodal interaction combines information from different modalities, such as gesture, voice, gaze, motion, or other sensor data, to improve the robustness and accuracy of intention recognition.

In this course project, we aim to improve a given multimodal user intention recognition model, especially under the following challenging conditions:

- Modal noise
- Missing modalities
- Cross-user testing scenario
- End-to-end raw data input and intention category output

---

## 3. Project Objectives

The main objectives of this project are:

1. Refactor the original feature-based model into an end-to-end model.
2. Use raw multimodal data as model input.
3. Use user A and user B as training data.
4. Use user C as testing data.
5. Construct modal noise baselines.
6. Construct missing modality baselines.
7. Improve the model under noise and missing modality conditions.
8. Organize experimental results for final report and defense.

---

## 4. Repository Structure

The planned repository structure is shown below:

```text
multimodal-intention-recognition/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── user_A/
│   │   ├── user_B/
│   │   └── user_C/
│   │
│   └── processed/
│
├── src/
│   ├── models/
│   └── modules/
│
├── experiments/
│
├── results/
│   ├── metrics/
│   ├── logs/
│   └── predictions/
│
├── figures/
│   ├── model_structure/
│   ├── curves/
│   └── result_charts/
│
├── checkpoints/
│   ├── baseline/
│   └── improved/
│
├── report/
│   ├── screenshots/
│   └── references/
│
└── docs/
````

---

## 5. Folder Description

### `data/`

This folder is used to store the dataset used in the project.

```text
data/
├── raw/
│   ├── user_A/
│   ├── user_B/
│   └── user_C/
└── processed/
```

* `raw/`: stores the original multimodal interaction data.
* `user_A/`: stores raw data from user A.
* `user_B/`: stores raw data from user B.
* `user_C/`: stores raw data from user C.
* `processed/`: stores processed or intermediate data if needed.

According to the project requirement, data from user A and user B will be used for training, while data from user C will be used for testing.

---

### `src/`

This folder will contain the main source code of the project.

```text
src/
├── models/
└── modules/
```

* `models/`: stores the baseline model and improved model structures.
* `modules/`: stores reusable functional modules, such as data preprocessing, modal noise processing, missing modality processing, and fusion modules.

At the current stage, only the folder structure is prepared. Specific code files will be added later.

---

### `experiments/`

This folder will contain experiment entry scripts.

It will be used to organize different experimental settings, including:

* Baseline experiment
* Modal noise baseline experiment
* Missing modality baseline experiment
* Improved model experiment
* Comparison experiments

Keeping experiment scripts in a separate folder makes the project easier to reproduce and manage.

---

### `results/`

This folder is used to save experiment outputs.

```text
results/
├── metrics/
├── logs/
└── predictions/
```

* `metrics/`: stores accuracy, loss, F1-score, and other evaluation results.
* `logs/`: stores training and testing logs.
* `predictions/`: stores model prediction results on the test set.

This folder will help organize the experimental evidence used in the final report.

---

### `figures/`

This folder is used to save figures for the project report and presentation.

```text
figures/
├── model_structure/
├── curves/
└── result_charts/
```

* `model_structure/`: stores model architecture diagrams.
* `curves/`: stores training loss curves and accuracy curves.
* `result_charts/`: stores comparison charts for noise and missing modality experiments.

These figures will be used in the final project report and defense slides.

---

### `checkpoints/`

This folder is used to save trained model weights.

```text
checkpoints/
├── baseline/
└── improved/
```

* `baseline/`: stores model checkpoints of the baseline model.
* `improved/`: stores model checkpoints of the improved model.

Large model files may not be uploaded to GitHub directly. If necessary, this folder can be ignored by `.gitignore`.

---

### `report/`

This folder is used to store materials related to the final project report.

```text
report/
├── screenshots/
└── references/
```

* `screenshots/`: stores screenshots proving that the end-to-end model runs successfully.
* `references/`: stores reference materials and citation lists.

The final report should include model design, improvement methods, experimental results, team contribution, and references.

---

### `docs/`

This folder is used to store project documentation.

Possible documents include:

* Project requirement summary
* Experiment plan
* Team contribution description
* Development notes
* Defense preparation materials

This folder helps the team maintain clear communication and project records during collaboration.

---

## 6. Development Plan

The project will be developed in several stages:

### Stage 1: Project Initialization

* Build the repository structure.
* Prepare project documentation.
* Organize the raw dataset according to user A, user B, and user C.

### Stage 2: Baseline Model Construction

* Refactor the original model workflow.
* Build an end-to-end training and testing pipeline.
* Use raw data as input and intention category as output.

### Stage 3: Baseline Experiments

* Train and test the baseline model.
* Record performance on the clean test set.
* Save logs, metrics, and initial results.

### Stage 4: Modal Noise Experiments

* Add noise to each single modality.
* Test noise levels of 20%, 40%, and 60%.
* Analyze the influence of modal noise on recognition accuracy.

### Stage 5: Missing Modality Experiments

* Remove one modality or two modalities.
* Train and test the model under different missing modality settings.
* Analyze model robustness under incomplete input conditions.

### Stage 6: Model Improvement

* Introduce improved fusion modules or additional loss terms.
* Improve recognition accuracy under noise and missing modality conditions.
* Compare the improved model with the baseline model.

### Stage 7: Report and Defense Preparation

* Organize experimental results.
* Draw result charts and model structure figures.
* Complete the final report.
* Prepare presentation slides for group defense.

---

## 7. Team Collaboration

This project will be completed by three team members.

The repository is designed to support collaborative work. Suggested task division includes:

* Dataset organization and preprocessing
* Baseline model refactoring
* Noise and missing modality experiments
* Model improvement
* Result analysis and visualization
* Report writing and defense preparation

The final contribution percentage of each member will be recorded in the project report.

---

## 8. Current Status

Current repository status:

* Project topic confirmed
* Repository structure designed
* Folder organization planned
* Code files not yet added
* Experiment implementation to be completed
* Report materials to be collected later

---

## 9. Notes

This repository is currently in the initialization stage. The project structure may be adjusted during later development according to the actual dataset format, model requirements, and experiment results.

```
```
