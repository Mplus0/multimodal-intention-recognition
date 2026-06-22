#!/usr/bin/env python3
import os
import json
import numpy as np
import librosa
from moviepy.editor import VideoFileClip
from datetime import datetime, timezone, timedelta
from tqdm import tqdm

# ============================
# 1. 配置 (适配 3.0 架构)
# ============================
# 基础路径
BASE_DIR = r"E:\smart AR\dataset"  # 视频文件路径保持不变
DATA_DIR = r"E:\smart AR\AR_Data_Process3.0\data"  # metadata 文件所在目录
SAVE_DATA_DIR = r"E:\smart AR\AR_Data_Process3.0\data\audio_features"  # 输出目录
os.makedirs(SAVE_DATA_DIR, exist_ok=True)

# 采样配置 (与 strong_gesture.py 严格同步)
AUDIO_WINDOW_SEC = 0.75  # 前后各 0.75s，总计 1.5s

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
# 2. 音频提取工具
# ============================
def extract_audio_from_mp4(mp4_path, wav_path):
    if os.path.exists(wav_path):
        return
    video = VideoFileClip(mp4_path)
    video.audio.write_audiofile(
        wav_path,
        fps=16000,
        nbytes=2,
        codec="pcm_s16le",
        verbose=False,
        logger=None
    )
    video.close()

def get_video_start_dt(video_name):
    """从文件名解析录制开始的 UTC datetime"""
    match_str = video_name.replace("interaction_", "").split('.')[0]
    dt = datetime.strptime(match_str, "%Y%m%d_%H%M%S")
    return dt.replace(tzinfo=timezone.utc)

# ============================
# 3. 核心 MFCC 提取 (39 维)
# ============================
def extract_mfcc_39d(wav_path, metadata_path, video_start_dt):
    """读取 Metadata 时间戳并提取 MFCC"""
    data = np.load(metadata_path, allow_pickle=True).item()
    approx_ts_list = data["approx_timestamps"]
    
    audio, sr = librosa.load(wav_path, sr=16000, mono=True)
    results = []

    for idx, ts_str in enumerate(approx_ts_list):
        # 1. 计算相对视频开始的秒数
        mid_dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        mid_sec = (mid_dt - video_start_dt).total_seconds()

        # 2. 确定 1.5s 窗口
        start = max(0.0, mid_sec - AUDIO_WINDOW_SEC)
        end = mid_sec + AUDIO_WINDOW_SEC
        
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = audio[start_sample:end_sample]

        # 过滤过短片段
        if len(segment) < sr * 0.1: continue

        # 3. 提取 MFCC (13维) + Delta + Delta2 = 39维
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        T = mfcc.shape[1]
        
        if T < 3:
            combined = np.vstack([mfcc, np.zeros_like(mfcc), np.zeros_like(mfcc)])
        else:
            width = min(9, T if T % 2 == 1 else T - 1)
            mfcc_delta = librosa.feature.delta(mfcc, width=width)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2, width=width)
            combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

        results.append({
            "id": idx,
            "timestamp": [float(start), float(end)],
            "feature": combined.T  # 转置为 (T, 39)
        })
    return results

# ============================
# 4. 主流程 (使用 VIDEO_NAMES)
# ============================
if __name__ == "__main__":
    print(f"准备处理 {len(VIDEO_NAMES)} 个视频的音频特征...")
    print(f"读取 metadata 路径: {DATA_DIR}")
    print(f"输出音频特征路径: {SAVE_DATA_DIR}")
    
    # 检查 VIDEO_NAMES 是否为空
    if not VIDEO_NAMES:
        print("⚠️ VIDEO_NAMES 列表为空，请添加要处理的视频")
        exit()

    # 遍历 VIDEO_NAMES 中的每个视频
    for video_name in VIDEO_NAMES:
        # 从视频名提取基础名称（去掉.mp4后缀）
        mp4_base = os.path.splitext(video_name)[0]
        
        # 构建对应的 metadata 文件路径 (从 DATA_DIR 读取)
        meta_file = f"metadata_strong_gesture_{mp4_base}.npy"
        mp4_path = os.path.join(BASE_DIR, video_name)
        meta_full_path = os.path.join(DATA_DIR, meta_file)  # 从 DATA_DIR 读取
        
        print(f"\n>>> 正在提取音频特征: {video_name}")

        # 检查视频文件是否存在
        if not os.path.exists(mp4_path):
            print(f"  [错误] 找不到视频文件: {mp4_path}")
            continue
            
        # 检查 metadata 文件是否存在
        if not os.path.exists(meta_full_path):
            print(f"  [错误] 找不到 metadata 文件: {meta_file}")
            print(f"  请先运行 strong_gesture.py 生成该视频的 metadata")
            continue
        
        # 临时wav文件放在 SAVE_DATA_DIR
        wav_path = os.path.join(SAVE_DATA_DIR, f"{mp4_base}.wav")
        out_npy_path = os.path.join(SAVE_DATA_DIR, f"audio_features_{mp4_base}.npy")
        out_json_path = os.path.join(SAVE_DATA_DIR, f"audio_features_{mp4_base}.json")

        # 1. 音频分离
        extract_audio_from_mp4(mp4_path, wav_path)

        # 2. 提取特征
        try:
            start_dt = get_video_start_dt(video_name)
            audio_results = extract_mfcc_39d(wav_path, meta_full_path, start_dt)
            
            if audio_results:
                # 3. 保存结果
                np.save(out_npy_path, audio_results)
                
                json_meta = [
                    {"id": r["id"], "timestamp": r["timestamp"], "feature_shape": list(r["feature"].shape)} 
                    for r in audio_results
                ]
                with open(out_json_path, "w", encoding="utf-8") as f:
                    json.dump(json_meta, f, indent=4)
                    
                print(f"✅ 完成！提取了 {len(audio_results)} 组音频特征序列")
            else:
                print(f"⚠️ 警告：未提取到任何有效的音频片段")
                
        except Exception as e:
            print(f"❌ 处理失败: {e}")
        
        # 可选：删除临时wav文件以节省空间
        if os.path.exists(wav_path):
            os.remove(wav_path)

    print(f"\n🎉 所有音频特征提取完成！保存目录: {SAVE_DATA_DIR}")