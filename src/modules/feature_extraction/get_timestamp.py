#!/usr/bin/env python3
import os
import numpy as np
import webrtcvad
import contextlib
import wave
import re
import json
from moviepy import editor as mp_edit
from datetime import datetime, timedelta, timezone

# ============================
# 1. 路径配置
# ============================
VIDEO_DIR = r"E:\smart AR\dataset"
SAVE_DIR = r"E:\smart AR\AR_Data_Process3.0\data"
os.makedirs(SAVE_DIR, exist_ok=True)

VIDEO_LABELS = {
    # =========================== office ==============================
    "interaction_20260306_072344.mp4": 0, # 菜单     # Bian
    "interaction_20260227_122606.mp4": 1, # 选择
    "interaction_20260227_122952.mp4": 2, # 放大
    "interaction_20260227_123354.mp4": 3, # 缩小
    "interaction_20260227_124559.mp4": 4, # 画笔
    "interaction_20260227_123745.mp4": 5, # 取消

    "interaction_20260131_120024.mp4": 0, # 菜单    # Luo
    "interaction_20260227_132951.mp4": 1, # 选择
    "interaction_20260227_133408.mp4": 2, # 放大
    "interaction_20260131_114156.mp4": 3, # 缩小
    "interaction_20260131_115150.mp4": 4, # 画笔
    "interaction_20260131_114852.mp4": 5, # 取消

    "interaction_20260301_073041.mp4": 0, # 菜单     # Gu
    "interaction_20260301_064753.mp4": 1, # 选择
    "interaction_20260306_072721.mp4": 2, # 放大
    "interaction_20260301_071948.mp4": 3, # 缩小
    "interaction_20260131_121548.mp4": 3, # 缩小
    "interaction_20260301_073435.mp4": 4, # 画笔
    "interaction_20260301_072503.mp4": 5, # 取消

    # ======================== museum ==================================
    # Luo 2026-01-31
    "interaction_20260131_071552.mp4": 0,  # 菜单
    "interaction_20260131_072412.mp4": 1,  # 选择
    "interaction_20260131_084300.mp4": 1,  # 选择
    "interaction_20260131_085611.mp4": 2,  # 放大
    "interaction_20260131_090139.mp4": 3,  # 缩小
    "interaction_20260131_085207.mp4": 4,  # 画笔
    "interaction_20260131_084732.mp4": 5,  # 取消

    # Gu
    "interaction_20260131_090917.mp4": 0,  # 菜单
    "interaction_20260131_090541.mp4": 1,  # 选择
    "interaction_20260131_065459.mp4": 2,  # 放大
    "interaction_20260131_070722.mp4": 3,  # 缩小
    "interaction_20260131_091657.mp4": 4,  # 画笔
    "interaction_20260131_091249.mp4": 5,  # 取消
    
    # Bian 
    "interaction_20260306_082346.mp4": 2,  # 放大
    "interaction_20260306_083107.mp4": 3,  # 缩小
    "interaction_20260306_083434.mp4": 1,  # 选择
    "interaction_20260306_084406.mp4": 0,  # 菜单
    "interaction_20260306_084853.mp4": 5,  # 取消
    "interaction_20260306_085830.mp4": 4,  # 画笔
    "interaction_20260306_090441.mp4": 1,  # 选择
}

CLASS_NAMES = ["menu", "select", "magnify", "narrow", "brush", "cancel"]

# ============================
# 2. 音频处理工具 (保持 10ms 级采样)
# ============================
def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        sr = wf.getframerate()
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        if wf.getnchannels() == 2:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
        return audio, sr

def vad_segments_refined(audio_path):
    """
    RMS 能量过滤 + VAD 逻辑：
    使用 70% 分位能量点作为基准，有效过滤环境白噪音。
    """
    audio, sr = read_wave(audio_path)
    vad = webrtcvad.Vad(2) 
    
    audio_float = audio.astype(np.float32)
    q70_rms = np.percentile(np.abs(audio_float), 70)
    print(f"   [Debug] 能量参考值 (70%分位): {q70_rms:.2f}")
    
    frame_ms = 30
    n = int(sr * (frame_ms / 1000.0))
    frames = [audio[i:i+n] for i in range(0, len(audio) - n, n)]
    
    times = []
    start = None
    detected_count = 0

    for idx, frm in enumerate(frames):
        is_speech = vad.is_speech(frm.tobytes(), sr)
        frm_rms = np.sqrt(np.mean(frm.astype(np.float32)**2))

        # 能量判定：语音帧需达到参考值的 0.8 倍
        valid_speech = is_speech and (frm_rms > q70_rms * 0.6)
        
        t = idx * (frame_ms / 1000.0)
        if valid_speech and start is None:
            start = t
        elif not valid_speech and start is not None:
            duration = t - start
            # 时长限制：过滤短促杂音
            if 0.3 <= duration <= 5.0:
                # 记录起止时间，保留 3 位小数确保毫秒精度
                times.append((round(max(0, start - 0.4), 3), round(t + 0.4, 3)))
                detected_count += 1
            start = None

    print(f"   [Debug] 能量过滤后剩余片段数: {detected_count}")
    return times

# ============================
# 3. 时间戳生成逻辑 (升级：支持毫秒)
# ============================
def get_iso_timestamps_ms(video_name, segments):
    """
    将视频内相对时间转换为带毫秒的 UTC ISO 字符串
    """
    match = re.search(r'interaction_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', video_name)
    if match:
        parts = [int(p) for p in match.groups()]
        start_dt = datetime(*parts, tzinfo=timezone.utc)
    else:
        start_dt = datetime.now(timezone.utc)

    iso_ts_list = []
    for s, e in segments:
        mid_sec = (s + e) / 2
        mid_dt = start_dt + timedelta(seconds=mid_sec)
        
        # 使用 %f 获取微秒，取前三位得到毫秒
        ms_str = mid_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        iso_ts_list.append(ms_str)
    return iso_ts_list

# ============================
# 4. 执行流程
# ============================
if __name__ == "__main__":
    print(f"🚀 [本地模式] 开始生成黄金时间戳...")
    print(f"🚀 [毫秒精度模式] 开始处理...")
    
    for video_name, label in VIDEO_LABELS.items():
        video_path = os.path.join(VIDEO_DIR, video_name)
        if not os.path.exists(video_path): continue
            
        print(f"\n>>> 处理: {video_name}")
        temp_wav = os.path.join(SAVE_DIR, "temp_vad_local.wav")
        
        try:
            clip = mp_edit.VideoFileClip(video_path)
            clip.audio.write_audiofile(temp_wav, fps=16000, codec="pcm_s16le", verbose=False, logger=None)
            clip.close()
            
            segments = vad_segments_refined(temp_wav)
            # 获取毫秒级中点时间戳
            iso_timestamps = get_iso_timestamps_ms(video_name, segments)
            
            if iso_timestamps:
                file_base = os.path.splitext(video_name)[0]
                
                # 1. 保存 .npy (供 strong_gesture.py 读取)
                np.save(os.path.join(SAVE_DIR, f"features_timestamp_{file_base}.npy"), {
                    "features": np.array([]),
                    "labels": np.full(len(iso_timestamps), label),
                    "video_names": np.array([video_name] * len(iso_timestamps)),
                    "approx_timestamps": iso_timestamps
                })

                # 2. 保存 .json (供人工核对相对时间)
                seg_info = {str(i): {"range": seg, "mid_utc": iso_timestamps[i]} 
                            for i, seg in enumerate(segments)}
                with open(os.path.join(SAVE_DIR, f"segments_info_{file_base}.json"), 'w') as f:
                    json.dump(seg_info, f, indent=4)
                
                print(f"   💾 已生成毫秒级 NPY 和 JSON 索引")
                
        except Exception as e:
            print(f"   ❌ 出错: {e}")
        finally:
            if os.path.exists(temp_wav): os.remove(temp_wav)

    print(f"\n🎉 处理完成！")