#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# ============================
# 1. 配置 (适配 3.0 架构)
# ============================
BASE_DIR = r"E:\smart AR\dataset"
IMU_PATH = os.path.join(BASE_DIR, "imu.csv") 

# 这里的路径存放 strong_gesture.py 生成的 metadata_...npy
INPUT_META_DIR = r"E:\smart AR\AR_Data_Process3.0\data"

# 这里的路径存放本脚本生成的 imu_features_...npy 和 .json
OUTPUT_IMU_DIR = r"E:\smart AR\AR_Data_Process3.0\data\imu_features"
os.makedirs(OUTPUT_IMU_DIR, exist_ok=True)

# 采样配置 (必须与 strong_gesture.py 严格一致)
TARGET_FRAME_NUM = 10      # 采样 10 帧
WINDOW_SEC = 0.75          # 前后各 0.75s

# 需要处理的头显视频列表
VIDEO_NAMES = [
    # =========================== office ==============================
    "interaction_20260306_072344.mp4",    # Bian
    "interaction_20260227_122606.mp4",    
    "interaction_20260227_122952.mp4",    
    "interaction_20260227_123354.mp4",    
    "interaction_20260227_124559.mp4",    
    "interaction_20260227_123745.mp4",
    
    "interaction_20260131_120024.mp4",    # Luo
    "interaction_20260227_132951.mp4",    
    "interaction_20260227_133408.mp4",     
    "interaction_20260131_114156.mp4",    
    "interaction_20260131_115150.mp4",    
    "interaction_20260131_114852.mp4",
    
    "interaction_20260301_073041.mp4",    # Gu
    "interaction_20260301_064753.mp4",    
    "interaction_20260306_072721.mp4",    
    "interaction_20260301_071948.mp4",
    "interaction_20260131_121548.mp4",    
    "interaction_20260301_073435.mp4",    
    "interaction_20260301_072503.mp4",

    # ======================== museum ==================================
    # Luo
    "interaction_20260131_071552.mp4",
    "interaction_20260131_072412.mp4",
    "interaction_20260131_084300.mp4",
    "interaction_20260131_084732.mp4",
    "interaction_20260131_085207.mp4",
    "interaction_20260131_085611.mp4",
    "interaction_20260131_090139.mp4",

    # Gu
    "interaction_20260131_065459.mp4",
    "interaction_20260131_070722.mp4",
    "interaction_20260131_090541.mp4",
    "interaction_20260131_090917.mp4",
    "interaction_20260131_091249.mp4",
    "interaction_20260131_091657.mp4",

    # Bian
    "interaction_20260306_082346.mp4",
    "interaction_20260306_083107.mp4",
    "interaction_20260306_083434.mp4",
    "interaction_20260306_084406.mp4",
    "interaction_20260306_084853.mp4",
    "interaction_20260306_085830.mp4",
    "interaction_20260306_090441.mp4",
]

# ============================
# 2. 预处理函数 (完全保留原动力学逻辑)
# ============================
def load_and_preprocess_imu(imu_csv_path):
    if not os.path.exists(imu_csv_path):
        raise FileNotFoundError(f"IMU 文件不存在: {imu_csv_path}")

    print(f"读取并清洗 IMU 数据...")
    df = pd.read_csv(imu_csv_path).dropna().sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # df["timestamp_sec"] = df["timestamp"].view(np.int64) / 1e9 # 纳秒转秒
    df["timestamp_sec"] = df["timestamp"].astype(np.int64) / 1e9

    # 四元数转欧拉角 (roll, pitch, yaw)
    r = R.from_quat(df[["rot_x", "rot_y", "rot_z", "rot_w"]].values)
    df[["roll", "pitch", "yaw"]] = r.as_euler("xyz", degrees=True)

    # 计算速度和加速度 (差分)
    dt = df["timestamp_sec"].diff().fillna(0.01).clip(lower=0.001)
    for axis in ["x", "y", "z"]:
        df[f"vel_{axis}"] = df[f"pos_{axis}"].diff() / dt
        df[f"acc_{axis}"] = df[f"vel_{axis}"].diff() / dt

    return df.fillna(0)

def sample_imu_segment(seg_df, target_frame_num=15):
    """ 线性插值重采样至 10 帧 """
    if seg_df.empty: 
        return np.zeros((target_frame_num, 12))
    
    # 提取 12 个通道
    cols = ["pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw",
            "vel_x", "vel_y", "vel_z", "acc_x", "acc_y", "acc_z"]
    features = seg_df[cols].values
    
    n_orig = len(features)
    if n_orig < 2: # 样本太少无法插值
        return np.tile(features[0] if n_orig == 1 else np.zeros(12), (target_frame_num, 1))

    new_idx = np.linspace(0, n_orig - 1, target_frame_num)
    sampled = np.zeros((target_frame_num, 12))
    for i in range(12):
        sampled[:, i] = np.interp(new_idx, np.arange(n_orig), features[:, i])
    return sampled

# ============================
# 3. 核心提取逻辑 (适配 Metadata 输入)
# ============================
def extract_imu_for_metadata(df_imu, metadata_path):
    """
    根据 metadata_strong_gesture_*.npy 中的时间戳提取 IMU 片段
    """
    meta_payload = np.load(metadata_path, allow_pickle=True).item()
    approx_tss = meta_payload["approx_timestamps"]
    labels = meta_payload["labels"]
    video_names = meta_payload["video_names"]
    
    valid_feats, valid_tss, valid_lbs, imu_ids = [], [], [], []

    for idx, ts_str in enumerate(approx_tss):
        mid_dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        mid_abs = mid_dt.timestamp()

        start, end = mid_abs - WINDOW_SEC, mid_abs + WINDOW_SEC
        seg_df = df_imu[(df_imu["timestamp_sec"] >= start) & (df_imu["timestamp_sec"] <= end)]
        
        # 重采样为 10 帧
        feat = sample_imu_segment(seg_df, TARGET_FRAME_NUM)
        
        valid_feats.append(feat)
        valid_tss.append(ts_str)
        valid_lbs.append(labels[idx])
        imu_ids.append(idx) # 记录 ID

    features_array = np.array(valid_feats)
    if len(features_array) > 0:
        mean = np.mean(features_array, axis=(0, 1), keepdims=True)
        std = np.std(features_array, axis=(0, 1), keepdims=True) + 1e-8
        features_array = (features_array - mean) / std

    return {
        "imu_id": np.array(imu_ids),            #  imu_id 键
        "approx_timestamps": valid_tss,        # (N,)
        "features": features_array,            # (N, 10, 12)
        "labels": np.array(valid_lbs),         # (N,)   标签，继承自上一级的文件
        # "video_names": video_names,            # (N,)
        # "features_shape": list(features_array.shape) if len(features_array) > 0 else []
    }

# ============================
# 4. 执行流程
# ============================
if __name__ == "__main__":
    df_imu_global = load_and_preprocess_imu(IMU_PATH)

    # ===== 修改：直接使用 VIDEO_NAMES 列表 =====
    print(f"准备处理 {len(VIDEO_NAMES)} 个视频的 IMU 数据...")
    
    # 检查 VIDEO_NAMES 是否为空
    if not VIDEO_NAMES:
        print("⚠️ VIDEO_NAMES 列表为空，请添加要处理的视频")
        exit()

    # 遍历 VIDEO_NAMES 中的每个视频
    for video_name in VIDEO_NAMES:
        # 从视频名提取基础名称（去掉.mp4后缀）
        name_base = os.path.splitext(video_name)[0]
        
        # 构建对应的 metadata 文件路径
        meta_file = f"metadata_strong_gesture_{name_base}.npy"
        meta_full_path = os.path.join(INPUT_META_DIR, meta_file)
        
        print(f"\n>>> 正在对齐 IMU 数据: {name_base} (视频: {video_name})")

        # 检查 metadata 文件是否存在
        if not os.path.exists(meta_full_path):
            print(f"  [错误] 找不到 metadata 文件: {meta_file}")
            print(f"  请先运行 strong_gesture.py 生成该视频的 metadata")
            continue
        
        # 提取 IMU 特征
        imu_result = extract_imu_for_metadata(df_imu_global, meta_full_path)

        if len(imu_result["features"]) > 0:
            # 保存到 OUTPUT_IMU_DIR
            out_npy_path = os.path.join(OUTPUT_IMU_DIR, f"imu_features_{name_base}.npy")
            np.save(out_npy_path, imu_result)
            
            out_json_path = os.path.join(OUTPUT_IMU_DIR, f"imu_features_{name_base}.json")
            json_meta = {
                "source_metadata": meta_file,
                "source_video": video_name,
                "target_video_base": name_base,
                "window_total_sec": WINDOW_SEC * 2,
                "frames_per_segment": TARGET_FRAME_NUM,
                "feature_channels": ["pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw",
                                     "vel_x", "vel_y", "vel_z", "acc_x", "acc_y", "acc_z"],
                "total_segments": len(imu_result["imu_id"])
            }
            with open(out_json_path, "w") as f:
                json.dump(json_meta, f, indent=4)
                
            print(f"✅ 完成！提取了 {len(imu_result['imu_id'])} 组 {TARGET_FRAME_NUM} 帧 IMU 序列")
        else:
            print(f"⚠️ 警告：未提取到任何有效的 IMU 片段")

    print(f"\n🎉 IMU 数据对齐任务全部完成！结果保存在: {OUTPUT_IMU_DIR}")
