# 智能眼镜多模态交互意图识别代码说明

## 目录结构

### 1. 模型训练和测试代码重构
- `train_and_test.py`
  - 基于锚点主干与残差辅助的多模态用户交互意图模型代码
  - 将这个脚本拆分成train.py和test.py，将特征提取代码整合进train.py和test.py代码，调试，实现端到端输入输出；重构代码后，运行train.py脚本进行模型训练，运行test.py脚本进行测试。

模型训练完成会保存成以下文件：
- `*.pt`：训练完成后的模型权重文件
- `scalers.pkl`：输入特征标准化器
- `label_encoder.pkl`：类别标签编码器

### 2. 特征提取代码，位于feature_extraction文件夹
- `get_timestamp.py`：提取时间戳
- `strong_gesture2.0.py`：提取手势特征
- `mfcc.py`：提取音频特征
- `ASR.py`：语音转文本特征
- `imu.py`：提取IMU特征

### 3. 数据集
- 位于dataset文件夹，包含3个用户在office场景和museum场景的6种意图交互数据。
- fisheye文件夹为鱼眼摄像头获取的视频，用作场景模态和手势模态数据。
- HoloLens文件夹为头显设备获取的视频，用作音频模态数据。
- imu.csv文件为IMU数据，用作IMU模态数据。根据交互视频的时间获取相应的IMU数据。
- AR_Data_Process3.0文件夹为各模态特征提取脚本。在重构代码时，在你新建的train.py和test.py中调用特征提取脚本。
- 训练集：
   # ================ office ===============
     "interaction_20260131_120024.mp4",    # Luo
     "interaction_20260227_132951.mp4",    
     "interaction_20260227_133408.mp4",     
     "interaction_20260131_114156.mp4",    
     "interaction_20260131_115150.mp4",    
     "interaction_20260131_114852.mp4"
    
     "interaction_20260301_073041.mp4",    # Gu
     "interaction_20260301_064753.mp4",    
     "interaction_20260306_072721.mp4",    
     "interaction_20260301_071948.mp4",    
     "interaction_20260131_121548.mp4",
     "interaction_20260301_073435.mp4",    
     "interaction_20260301_072503.mp4"

    # ============== museum ===============
    "interaction_20260131_071552.mp4",      # Luo
    "interaction_20260131_072412.mp4",
    "interaction_20260131_084300.mp4",
    "interaction_20260131_084732.mp4",
    "interaction_20260131_085207.mp4",
    "interaction_20260131_085611.mp4",
    "interaction_20260131_090139.mp4",

    "interaction_20260131_065459.mp4",       # Gu
    "interaction_20260131_070722.mp4",
    "interaction_20260131_090541.mp4",
    "interaction_20260131_090917.mp4",
    "interaction_20260131_091249.mp4",
    "interaction_20260131_091657.mp4",

- 测试集：
    # ============== office ================
    "interaction_20260306_072344.mp4",    # Bian
    "interaction_20260227_122606.mp4",    
    "interaction_20260227_122952.mp4",    
    "interaction_20260227_123354.mp4",    
    "interaction_20260227_124559.mp4",    
    "interaction_20260227_123745.mp4"

    # ============= museum ==============
    "interaction_20260306_082346.mp4"   # Bian
    "interaction_20260306_083107.mp4"
    "interaction_20260306_083434.mp4"
    "interaction_20260306_084406.mp4"
    "interaction_20260306_084853.mp4"
    "interaction_20260306_085830.mp4"
    "interaction_20260306_090441.mp4"

### 4. 课程项目代码提交
- 提交小组完整代码压缩文件，包括已训练模型文件。在报告中汇报模型训练过程和测试过程、模型效果、代码结构、代码运行方式等。汇报总训练时间、平均样本训练时间、平均样本测试等。