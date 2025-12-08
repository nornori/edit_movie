import os
import glob
import argparse
import urllib.parse
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import cv2
import librosa
import torch
import mediapipe as mp
import easyocr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# ==========================================
# 設定
# ==========================================
FPS_ANALYSIS = 10.0      # 0.1秒刻み
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_OCR = True           # 文字認識を行うか（重いですがONにします）

# ==========================================
# 1. XML解析 (Ground Truth)
# ==========================================
def clean_xml_path(raw_url):
    if not raw_url: return ""
    decoded = urllib.parse.unquote(raw_url).replace("file://localhost/", "").replace("file://", "")
    if os.name == 'nt' and decoded.startswith("/") and ':' in decoded:
        decoded = decoded.lstrip("/")
    return decoded.replace("/", "\\") if os.name == 'nt' else decoded

def parse_xml_ground_truth(xml_path, fps_base=59.94):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return [], None

    video_path = None
    timeline_events = []

    for path_node in root.iter("pathurl"):
        p = clean_xml_path(path_node.text)
        if p.lower().endswith(('.mp4', '.mov', '.mkv', '.avi')):
            if os.path.exists(p):
                video_path = p
                break
            local = os.path.join(os.path.dirname(xml_path), os.path.basename(p))
            if os.path.exists(local):
                video_path = local
                break
    
    if not video_path:
        return [], None

    for seq in root.iter("sequence"):
        for track in seq.findall("./media/video/track"):
            for clip in track.findall("clipitem"):
                start = int(clip.find("start").text)
                end = int(clip.find("end").text)
                if start < 0 or end < 0: continue
                
                t_start = start / fps_base
                t_end = end / fps_base
                
                attrs = {
                    "start": t_start, "end": t_end,
                    "is_used": 1,
                    "scale": 100.0, "pos_x": 0.5, "pos_y": 0.5,
                    "graphic": 0, "broll": 0
                }
                
                for filter_node in clip.findall("filter"):
                    eff = filter_node.find("effect")
                    if eff is None: continue
                    for p in eff.findall("parameter"):
                        pid = p.find("parameterid").text.lower() if p.find("parameterid") is not None else ""
                        val = p.find("value").text if p.find("value") is not None else "0"
                        try:
                            v_float = float(val)
                            if "scale" in pid: attrs["scale"] = v_float
                        except: pass
                timeline_events.append(attrs)

    return timeline_events, video_path

# ==========================================
# 2. 特徴量抽出 (Full Features)
# ==========================================
class FeatureExtractor:
    def __init__(self):
        print("[1/5] Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        
        print("[2/5] Loading MediaPipe FaceMesh...")
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

        print("[3/5] Initializing Saliency Detector...")
        # OpenCV Contribが必要です
        try:
            self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        except AttributeError:
            print("Warning: cv2.saliency not found. Install opencv-contrib-python. Using dummy saliency.")
            self.saliency = None

        if USE_OCR:
            print("[4/5] Initializing OCR (EasyOCR)...")
            # GPUがあれば使い、なければCPU
            self.reader = easyocr.Reader(['en', 'ja'], gpu=(DEVICE=="cuda"))
        
        print("[5/5] All models loaded.")

    def get_audio_features(self, video_path, duration_sec):
        """音声特徴量 (RMS, 無音, 話者ID)"""
        print("Extracting Audio & Diarization...")
        y, sr = librosa.load(video_path, sr=16000, mono=True)
        target_len = int(duration_sec * FPS_ANALYSIS)
        
        # 1. RMS & Speaking
        hop_length = int(sr / FPS_ANALYSIS)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        
        if len(rms) < target_len:
            rms = np.pad(rms, (0, target_len - len(rms)))
        else:
            rms = rms[:target_len]
        is_speaking = (rms > 0.01).astype(int)

        # 2. Silence Duration
        silence_durations = []
        current_silence = 0.0
        step_ms = 1000.0 / FPS_ANALYSIS
        for spk in is_speaking:
            if spk == 0: current_silence += step_ms
            else: current_silence = 0.0
            silence_durations.append(current_silence)

        # 3. Speaker ID (Clustering based on MFCC)
        # 簡易的な話者分離: MFCCを計算してK-Meansでクラスタリングする
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        mfcc = mfcc.T[:target_len] # (Time, Features)
        
        # 長さが足りない場合の埋め合わせ
        if mfcc.shape[0] < target_len:
             padding = np.zeros((target_len - mfcc.shape[0], 13))
             mfcc = np.vstack([mfcc, padding])
        
        # 声が出ている部分だけでクラスタリング（ノイズ除去）
        valid_indices = is_speaking == 1
        speaker_ids = np.zeros(target_len, dtype=int)
        
        if np.sum(valid_indices) > 10: # ある程度喋っていれば
            try:
                scaler = StandardScaler()
                mfcc_scaled = scaler.fit_transform(mfcc[valid_indices])
                # 2話者と仮定して分類 (話者が多い場合はn_clustersを増やす)
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                labels = kmeans.fit_predict(mfcc_scaled)
                speaker_ids[valid_indices] = labels + 1 # ID: 1, 2... (0は無音)
            except:
                pass # エラー時は全員ID:0

        return rms, is_speaking, np.array(silence_durations), speaker_ids

    def get_visual_features(self, video_path):
        """映像特徴量 (CLIP, Face, Motion, Scene, Saliency, OCR)"""
        print("Extracting Visual Features (Deep Analysis)...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        data_records = []
        step = int(fps / FPS_ANALYSIS)
        if step < 1: step = 1
        
        prev_gray = None
        prev_hist = None
        
        # OCRは重いので1秒に1回程度にするためのカウンタ
        ocr_interval = int(fps) 
        last_ocr_text_active = 0
        last_ocr_word_count = 0
        
        for i in tqdm(range(0, total_frames, step)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
            
            time_sec = i / fps
            
            # Preprocessing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (160, 90))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            # --- 1. Motion & Scene Change ---
            motion = 0.0
            scene_change_score = 0.0
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

            if prev_gray is not None:
                motion = np.mean(cv2.absdiff(gray_small, prev_gray))
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                scene_change_score = 1.0 - correlation
                if scene_change_score < 0: scene_change_score = 0
            prev_gray = gray_small
            prev_hist = hist
            
            # --- 2. Saliency Map (視線誘導) ---
            sal_x, sal_y = 0.5, 0.5 # Default Center
            if self.saliency is not None:
                success, saliency_map = self.saliency.computeSaliency(gray)
                if success:
                    # 重心(Moment)を計算して「視線の中心」とする
                    M = cv2.moments(saliency_map)
                    if M["m00"] != 0:
                        sal_x = (M["m10"] / M["m00"]) / w
                        sal_y = (M["m01"] / M["m00"]) / h

            # --- 3. OCR (Text Detection) ---
            # 毎フレームやると遅すぎるので、1秒に1回更新し、間は前の値を保持
            if USE_OCR and (i % ocr_interval < step):
                try:
                    # 簡易化のため英語モードでも数字やアルファベットは取れる
                    results = self.reader.readtext(frame, detail=0) 
                    last_ocr_text_active = 1 if results else 0
                    last_ocr_word_count = len(results) # 単語数/行数
                except:
                    pass
            
            # --- 4. Face ---
            face_feats = {
                "face_count": 0, "face_center_x": 0.0, "face_center_y": 0.0,
                "face_size": 0.0, "face_eyebrow_raise": 0.0, "face_mouth_open": 0.0
            }
            results = self.mp_face_mesh.process(rgb)
            if results.multi_face_landmarks:
                face_feats["face_count"] = 1
                lm = results.multi_face_landmarks[0]
                face_feats["face_center_x"] = lm.landmark[1].x
                face_feats["face_center_y"] = lm.landmark[1].y
                upper = np.array([lm.landmark[13].x, lm.landmark[13].y])
                lower = np.array([lm.landmark[14].x, lm.landmark[14].y])
                face_feats["face_mouth_open"] = np.linalg.norm(upper - lower) * 100
                
            # --- 5. CLIP ---
            pil_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Original resolution for CLIP
            inputs = self.clip_processor(images=pil_img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            clip_vec = image_features.cpu().numpy()[0]
            
            # Record
            record = {
                "time": round(time_sec, 2),
                "visual_motion": motion,
                "scene_change": scene_change_score,
                "saliency_x": sal_x,    # ★復活
                "saliency_y": sal_y,    # ★復活
                "text_is_active": last_ocr_text_active, # ★復活
                "text_word": last_ocr_word_count,       # ★復活
                **face_feats
            }
            for idx, val in enumerate(clip_vec):
                record[f"clip_{idx}"] = val
                
            data_records.append(record)
            
        cap.release()
        return pd.DataFrame(data_records), duration

# ==========================================
# メイン処理
# ==========================================
def process_xml_to_csv(xml_file):
    print(f"\nProcessing: {xml_file}")
    
    gt_events, video_path = parse_xml_ground_truth(xml_file)
    if not gt_events:
        print("Skipping (No data or video found).")
        return

    extractor = FeatureExtractor()
    
    # 映像解析
    df_visual, duration = extractor.get_visual_features(video_path)
    
    # 音声解析 (話者ID含む)
    rms, is_speaking, silence_dur, speaker_ids = extractor.get_audio_features(video_path, duration)
    
    # 長さ合わせ
    min_len = min(len(df_visual), len(rms))
    df_visual = df_visual.iloc[:min_len]
    df_visual["audio_energy_rms"] = rms[:min_len]
    df_visual["audio_is_speaking"] = is_speaking[:min_len]
    df_visual["silence_duration_ms"] = silence_dur[:min_len]
    df_visual["speaker_id"] = speaker_ids[:min_len] # ★復活
    
    # Target Mapping
    targets = []
    for t in df_visual["time"]:
        row_target = {
            "target_is_used": 0, "target_scale": 100.0, 
            "target_pos_x": 0.5, "target_pos_y": 0.5,
            "target_graphic": 0, "target_broll": 0
        }
        for ev in gt_events:
            if ev["start"] <= t < ev["end"]:
                row_target["target_is_used"] = 1
                row_target["target_scale"] = ev["scale"]
                break 
        targets.append(row_target)
        
    df_targets = pd.DataFrame(targets)
    df_final = pd.concat([df_visual, df_targets], axis=1)
    
    output_csv = xml_file.replace(".xml", "_dataset.csv")
    df_final.to_csv(output_csv, index=False)
    print(f"Dataset saved: {output_csv}")

if __name__ == "__main__":
    target_folder = r"C:\Users\yushi\Documents\プログラム\xmlai\edit_triaining"
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default=target_folder)
    args = parser.parse_args()
    
    xml_files = glob.glob(os.path.join(args.folder, "*.xml"))
    print(f"Found {len(xml_files)} XML files.")
    for f in xml_files:
        process_xml_to_csv(f)