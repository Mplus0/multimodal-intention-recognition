#!/usr/bin/env python3
import os
import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel
import cv2
from PIL import Image
import mediapipe as mp
import json
from tqdm import tqdm
from datetime import datetime, timedelta, timezone

# ==================== 1. 路径与参数配置 ====================
BASE_DIR = r"E:\smart AR\dataset"
INPUT_DATA_DIR = r"E:\smart AR\AR_Data_Process3.0\data" 
OUTPUT_DIR = r"E:\smart AR\AR_Data_Process3.0\data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLIP_MODEL_PATH = r"E:\smart AR\AR_Data_Process3.0\models\clip_teacher_model"

# 序列采样配置
SEQ_LEN = 10                 # 提取 10 帧
HALF_WINDOW_MS = 750        # 前后各 0.75 秒，总计 1.5 秒窗口

AVI_TO_MP4_MAP = {
     # =========================== office ==============================
    "Video_20260306_152340690.avi": "interaction_20260306_072344.mp4",  # 菜单      # Bian
    "Video_20260227_202553335.avi": "interaction_20260227_122606.mp4",  # 选择
    "Video_20260227_202953348.avi": "interaction_20260227_122952.mp4",  # 放大
    "Video_20260227_203348219.avi": "interaction_20260227_123354.mp4",  # 缩小
    "Video_20260227_204553897.avi": "interaction_20260227_124559.mp4",  # 画笔
    "Video_20260227_203753817.avi": "interaction_20260227_123745.mp4",  # 取消

    "Video_20260131_200029359.avi": "interaction_20260131_120024.mp4",  # 菜单      # Luo
    "Video_20260227_213001434.avi": "interaction_20260227_132951.mp4",  # 选择
    "Video_20260227_213404452.avi": "interaction_20260227_133408.mp4",  # 放大
    "Video_20260131_194205407.avi": "interaction_20260131_114156.mp4",  # 缩小
    "Video_20260131_195202906.avi": "interaction_20260131_115150.mp4",  # 画笔
    "Video_20260131_194854095.avi": "interaction_20260131_114852.mp4",  # 取消

    "Video_20260301_153037623.avi": "interaction_20260301_073041.mp4",  # 菜单     # Gu
    "Video_20260301_144803454.avi": "interaction_20260301_064753.mp4",  # 选择
    "Video_20260306_152721366.avi": "interaction_20260306_072721.mp4",  # 放大
    "Video_20260301_151942635.avi": "interaction_20260301_071948.mp4",  # 缩小
    "Video_20260131_201556629.avi": "interaction_20260131_121548.mp4",  # 缩小
    "Video_20260301_153434856.avi": "interaction_20260301_073435.mp4",  # 画笔
    "Video_20260301_152459131.avi": "interaction_20260301_072503.mp4",  # 取消

    # ======================== museum ==================================
    # Luo - 2026-01-31
    "Video_20260131_151559270.avi": "interaction_20260131_071552.mp4",  # 菜单
    "Video_20260131_152410916.avi": "interaction_20260131_072412.mp4",  # 选择
    "Video_20260131_164304016.avi": "interaction_20260131_084300.mp4",  # 选择
    "Video_20260131_164745532.avi": "interaction_20260131_084732.mp4",  # 取消
    "Video_20260131_165208524.avi": "interaction_20260131_085207.mp4",  # 画笔
    "Video_20260131_165614756.avi": "interaction_20260131_085611.mp4",  # 放大
    "Video_20260131_170142792.avi": "interaction_20260131_090139.mp4",  # 缩小

    # Gu
    "Video_20260131_145524524.avi": "interaction_20260131_065459.mp4",  # 放大
    "Video_20260131_150734369.avi": "interaction_20260131_070722.mp4",  # 缩小
    "Video_20260131_170539636.avi": "interaction_20260131_090541.mp4",  # 选择
    "Video_20260131_170919896.avi": "interaction_20260131_090917.mp4",  # 菜单
    "Video_20260131_171253889.avi": "interaction_20260131_091249.mp4",  # 取消
    "Video_20260131_171648040.avi": "interaction_20260131_091657.mp4",  # 画笔
    
    # Bian - 2026-03-06
    "Video_20260306_162401599.avi": "interaction_20260306_082346.mp4",  # 放大
    "Video_20260306_163105571.avi": "interaction_20260306_083107.mp4",  # 缩小
    "Video_20260306_163434878.avi": "interaction_20260306_083434.mp4",  # 选择
    "Video_20260306_164407883.avi": "interaction_20260306_084406.mp4",  # 菜单
    "Video_20260306_164902044.avi": "interaction_20260306_084853.mp4",  # 取消
    "Video_20260306_165839689.avi": "interaction_20260306_085830.mp4",  # 画笔
    "Video_20260306_170449073.avi": "interaction_20260306_090441.mp4",  # 选择
}

# ==================== 2. 模型加载 (针对 4090 优化) ====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

# 如果你只有单显卡，请尝试改为 cuda:0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 [设备] 当前使用: {device}")

clip_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_PATH, local_files_only=True)
clip_vision = CLIPVisionModel.from_pretrained(CLIP_MODEL_PATH, local_files_only=True).to(device).eval()

# ==================== 3. 核心处理函数 ====================

def crop_hand(img_pil):
    """ MediaPipe 裁剪逻辑 (保持 40% Padding) """
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        xs = [int(lm.x * w) for hand in results.multi_hand_landmarks for lm in hand.landmark]
        ys = [int(lm.y * h) for hand in results.multi_hand_landmarks for lm in hand.landmark]
        x1, y1, x2, y2 = max(0, min(xs)), max(0, min(ys)), min(w, max(xs)), min(h, max(ys))
        cw, ch = x2 - x1, y2 - y1
        pad = 0.4
        x1, y1 = max(0, int(x1 - cw * pad)), max(0, int(y1 - ch * pad))
        x2, y2 = min(w, int(x2 + cw * pad)), min(h, int(y2 + ch * pad))
        return img_pil.crop((x1, y1, x2, y2)).resize((224, 224), Image.LANCZOS)
    return img_pil.rotate(0).resize((224, 224), Image.LANCZOS)

@torch.no_grad()
def extract_clip_sequence(video_path, center_ms):
    """ 提取 10 帧特征序列，不再生成图片 """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_ms = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps) * 1000
    
    start_ms = center_ms - HALF_WINDOW_MS
    end_ms = center_ms + HALF_WINDOW_MS
    
    # 边界检查
    if start_ms < 0 or end_ms > total_ms:
        cap.release()
        return None

    seq_offsets = np.linspace(start_ms, end_ms, SEQ_LEN)
    seq_features = []

    for msec in seq_offsets:
        cap.set(cv2.CAP_PROP_POS_MSEC, msec)
        ok, frame = cap.read()
        if not ok: break
        
        # 预处理与特征提取
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hand_box = crop_hand(img_pil)
        
        inputs = clip_processor(images=hand_box.convert("RGB"), return_tensors="pt").to(device)
        outputs = clip_vision(**inputs)
        # 提取特征并压入列表
        feat = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        seq_features.append(feat)
    
    cap.release()
    return np.array(seq_features) if len(seq_features) == SEQ_LEN else None

# ==================== 时间计算 ====================
def get_avi_sync_ms(avi_path, utc_target):
    """ 时间对齐逻辑 """
    avi_fn = os.path.basename(avi_path)

    # 1. 提取AVI录制开始的本地时间 (精确到毫秒)
    # 示例: Video_20260301_153037623 -> 15:30:37.623
    time_part = avi_fn.split('_')[1] + "_" + avi_fn.split('_')[2].split('.')[0]

    # 2. 统一转为 UTC 时间 (本地时间 - 8小时)
    avi_utc_start = datetime.strptime(time_part, '%Y%m%d_%H%M%S%f') - timedelta(hours=8)
    
    # 3. 计算【动作发生时刻】减去【视频开始时刻】的差值
    # 这个差值就是动作在视频中的相对位置
    diff_ms = (utc_target - avi_utc_start).total_seconds() * 1000

    # 4. 获取视频物理属性
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {avi_fn}")
        return None
    
    # 5. 全包容判定
    # 只要 diff_ms >= 0 且不超过视频总长度，就是有效片段
    dur_ms = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)) * 1000
    cap.release()
    
    return diff_ms if 0 <= diff_ms <= dur_ms else None

# ==================== 4. 主执行流 ====================
if __name__ == "__main__":
    for avi_name, mp4_name in AVI_TO_MP4_MAP.items():
        mp4_base = os.path.splitext(mp4_name)[0]
        input_npy_path = os.path.join(INPUT_DATA_DIR, f"features_timestamp_{mp4_base}.npy")
        # 确保路径拼接正确
        video_full_path = os.path.join(BASE_DIR, avi_name)

        # 增加视频文件物理存在检查
        if not os.path.exists(video_full_path) or not os.path.exists(input_npy_path):
            print(f"❌ 跳过：视频文件不存在 -> {video_full_path} 或 找不到输入特征文件 {input_npy_path}")
            continue

        data_payload = np.load(input_npy_path, allow_pickle=True).item()
        ts_list = data_payload["approx_timestamps"]
        labels = data_payload["labels"]
        
        valid_feats, valid_lbs, valid_tss = [], [], []
        debug_log = {}

        print(f"\n>>> 正在处理: {avi_name} (目标帧数: {SEQ_LEN})")

        for i, ts_str in tqdm(enumerate(ts_list), total=len(ts_list)):
            try:
                utc_dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00')).replace(tzinfo=None)
                offset_ms = get_avi_sync_ms(video_full_path, utc_dt)

                if offset_ms is None: 
                    continue

                # 核心提取：10 帧序列
                sequence = extract_clip_sequence(video_full_path, offset_ms)
                
                if sequence is not None:
                    # 这一步保证了索引从 0 开始连续存放
                    valid_feats.append(sequence)
                    valid_lbs.append(labels[i])
                    valid_tss.append(ts_str)
                    
                    debug_log[str(len(valid_feats)-1)] = {
                        "original_idx": i,
                        "utc_time": ts_str,
                        "msec_center": round(offset_ms, 2)
                        # "shape": str(sequence.shape)
                    }
                else:
                    print(f"  [Skip] 片段 {i} 帧读取失败")

            except Exception as e:
                print(f"  ❌ 处理片段 {i} 时出错: {e}")

        # --- 数据保存 ---
        if valid_feats:
            # 1. 保存特征文件 (N, 10, 768)
            final_npy = {
                "features": np.array(valid_feats),    # (N, 10, Dim)
                "labels": np.array(valid_lbs),     # (N,)
                "video_names": np.array([mp4_name] * len(valid_lbs)), # (N,)
                "approx_timestamps": valid_tss     # (N,) Center UTC
            }
            np.save(os.path.join(OUTPUT_DIR, f"strong_gesture_features_{mp4_base}.npy"), final_npy)

            # 2. 保存 Metadata (格式与特征文件一致，仅去掉 features)
            meta_npy = {k: v for k, v in final_npy.items() if k != "features"}
            np.save(os.path.join(OUTPUT_DIR, f"metadata_strong_gesture_{mp4_base}.npy"), meta_npy)

            # 3. 保存 Debug JSON
            with open(os.path.join(OUTPUT_DIR, f"debug_strong_gesture_{mp4_base}.json"), 'w') as f:
                json.dump(debug_log, f, indent=4)
            
            print(f"    ✅ 保存完成！有效片段: {len(valid_feats)}/{len(ts_list)}")
        else:
            print("    ⚠️ 该视频未提取到任何有效时序片段")
            
    print(f"\n🎉 10帧强手势时序特征提取保存完成！")
    print(f"   输出目录: {OUTPUT_DIR}")
