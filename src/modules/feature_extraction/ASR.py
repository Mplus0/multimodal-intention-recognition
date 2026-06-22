#!/usr/bin/env python3
import os
import json
import numpy as np
import librosa
import whisper
from moviepy.editor import VideoFileClip
from sentence_transformers import SentenceTransformer
from pypinyin import pinyin, Style
from datetime import datetime, timezone

# ============================
# 1. 路径配置
# ============================
HOLOLENS_VIDEO_DIR = r"E:\smart AR\dataset"
# 这里指向包含 metadata_strong_gesture_...npy 的目录
DATA_DIR = r"E:\smart AR\AR_Data_Process3.0\data"

TEXT_OUT_DIR = os.path.join(DATA_DIR, "text_features")
os.makedirs(TEXT_OUT_DIR, exist_ok=True)

LOCAL_SENTENCE_MODEL = r"E:\smart AR\AR_Data_Process3.0\models\all-MiniLM-L6-v2"

# ============================
# 2. 核心参数
# ============================
GESTURE_TIMESTEPS = 10
# 你的要求：前后各 0.75s，总长度 1.5s
AUDIO_WINDOW_SEC = 0.75 

# 需要处理的头显视频列表
VIDEO_NAMES = [
    # # =========================== office ==============================
    "interaction_20260306_072344.mp4",    # Bian，测试集
    "interaction_20260227_122606.mp4",    
    "interaction_20260227_122952.mp4",    
    "interaction_20260227_123354.mp4",    
    "interaction_20260227_124559.mp4",    
    "interaction_20260227_123745.mp4",
    
    "interaction_20260131_120024.mp4",    # Luo，训练数据
    "interaction_20260227_132951.mp4",    
    "interaction_20260227_133408.mp4",     
    "interaction_20260131_114156.mp4",    
    "interaction_20260131_115150.mp4",    
    "interaction_20260131_114852.mp4",
    
    "interaction_20260301_073041.mp4",    # Gu，训练数据
    "interaction_20260301_064753.mp4",    
    "interaction_20260306_072721.mp4",    
    "interaction_20260301_071948.mp4",    
    "interaction_20260131_121548.mp4",
    "interaction_20260301_073435.mp4",    
    "interaction_20260301_072503.mp4",

    # # ======================== museum ==================================
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
# 3. 功能模块
# ============================

def extract_audio_from_mp4(mp4_path, wav_path):
    if os.path.exists(wav_path): return
    video = VideoFileClip(mp4_path)
    video.audio.write_audiofile(
        wav_path, fps=16000, nbytes=2, codec="pcm_s16le",
        ffmpeg_params=["-ac", "1"], verbose=False, logger=None
    )
    video.close()

def parse_metadata_timestamps(metadata_path, mp4_name, window_sec):
    """
    从 metadata 中读取毫秒级时间戳，并计算音频裁剪区间
    """
    data = np.load(metadata_path, allow_pickle=True).item()
    approx_ts = data["approx_timestamps"] # 这里的格式是 2026-03-01T07:30:41.425Z
    
    # 解析 MP4 起始时间 (用于计算相对偏移)
    name_parts = os.path.basename(mp4_name).split('_')
    dt_str = name_parts[1] + name_parts[2].split('.')[0]
    video_start_dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")

    time_ranges = []
    for ts_val in approx_ts:
        # 直接解析带毫秒的 ISO 字符串
        current_dt = datetime.fromisoformat(str(ts_val).replace('Z', '+00:00')).replace(tzinfo=None)
        
        # 计算相对视频起始点的秒数（保留毫秒精度）
        mid_sec = (current_dt - video_start_dt).total_seconds()
        
        start = max(0.0, mid_sec - window_sec)
        end = mid_sec + window_sec
        time_ranges.append((start, end))
    return time_ranges

def transcribe_and_embed(wav_path, time_ranges, whisper_model, st_model):
    audio, sr = librosa.load(wav_path, sr=16000, mono=True)
    segments_meta = []
    raw_embeddings = []

    print(f"  正在处理 {len(time_ranges)} 个片段...")
    for idx, (start, end) in enumerate(time_ranges):
        start_sample, end_sample = int(start * sr), int(end * sr)
        segment_audio = audio[start_sample:end_sample]

        # ASR 识别
        text = ""
        if len(segment_audio) >= sr * 0.1:
            try:
                # 使用 Whisper 进行识别
                result = whisper_model.transcribe(segment_audio, language="zh", fp16=False)
                text = result["text"].strip()
            except: text = ""

        # 转拼音
        pinyin_text = ""
        if text:
            pinyin_list = pinyin(text, style=Style.TONE)
            pinyin_text = " ".join([item[0] for item in pinyin_list])

        segments_meta.append({
            "id": idx,
            "abs_timestamp": [float(start), float(end)], # 这里存的是相对于视频开头的秒数
            "text": text,
            "pinyin": pinyin_text
        })
        
        # 语义编码 (384维)
        emb = st_model.encode(pinyin_text if pinyin_text else "", normalize_embeddings=True)
        raw_embeddings.append(emb)

    # 转换为 (N, 10, 384) 格式以适配下游模型
    embeddings_np = np.array(raw_embeddings) # (N, 384)
    text_features = np.tile(embeddings_np[:, np.newaxis, :], (1, GESTURE_TIMESTEPS, 1))
    
    return text_features, segments_meta

# ============================
# 4. 主流程
# ============================
if __name__ == "__main__":
    # ===== 修改：直接使用 VIDEO_NAMES 列表 =====
    print(f"准备处理 {len(VIDEO_NAMES)} 个视频...")
    
    # 检查 VIDEO_NAMES 是否为空
    if not VIDEO_NAMES:
        print("⚠️ VIDEO_NAMES 列表为空，请添加要处理的视频")
        exit()

    print("正在加载 Whisper (small) 与 SentenceTransformer 模型...")
    whisper_model = whisper.load_model("small")
    st_model = SentenceTransformer(LOCAL_SENTENCE_MODEL)

    # 遍历 VIDEO_NAMES 中的每个视频
    for video_name in VIDEO_NAMES:
        name_base = os.path.splitext(video_name)[0]
        
        # 构建对应的 metadata 文件路径
        meta_file = f"metadata_strong_gesture_{name_base}.npy"
        mp4_path = os.path.join(HOLOLENS_VIDEO_DIR, video_name)
        meta_path = os.path.join(DATA_DIR, meta_file)
        
        print(f"\n>>> 提取 ASR 特征 (基于过滤后数据): {video_name}")

        # 检查文件是否存在
        if not os.path.exists(mp4_path):
            print(f"  [错误] 找不到原始视频: {mp4_path}")
            continue
            
        if not os.path.exists(meta_path):
            print(f"  [错误] 找不到 metadata 文件: {meta_path}")
            print(f"  请先运行 strong_gesture.py 生成该文件的 metadata")
            continue
        
        # 1. 提取临时音频
        temp_wav = os.path.join(DATA_DIR, f"temp_asr_{name_base}.wav")
        extract_audio_from_mp4(mp4_path, temp_wav)

        # 2. 解析毫秒级时间戳 (由 metadata 提供)
        try:
            time_ranges = parse_metadata_timestamps(meta_path, video_name, AUDIO_WINDOW_SEC)
        except Exception as e:
            print(f"  [跳过] 解析失败: {e}")
            continue

        # 3. ASR + Embedding
        features, meta = transcribe_and_embed(temp_wav, time_ranges, whisper_model, st_model)

        # 4. 保存结果
        out_npy = os.path.join(TEXT_OUT_DIR, f"text_features_{name_base}.npy")
        out_json = os.path.join(TEXT_OUT_DIR, f"text_features_{name_base}.json")
        
        # 保存为最终特征文件
        np.save(out_npy, {"features": features.astype(np.float32), "metadata": meta})
        # 保存为可读 JSON 方便检查识别文本
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            
        print(f"  ✅ 成功保存 {len(meta)} 个文本语义样本")
        
        if os.path.exists(temp_wav): 
            try: os.remove(temp_wav)
            except: pass

    print("\n🎉 级联 ASR 任务处理完成！所有输出已与视觉特征对齐。")
