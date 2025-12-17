"""
動画編集推論パイプライン

動画ファイルを入力として、学習済みモデルで編集を予測し、
Premiere Pro XML形式で出力します。
"""
import argparse
import torch
import numpy as np
import pandas as pd
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple

from src.model.model import create_model
from src.model.model_persistence import load_model
from src.training.multimodal_preprocessing import AudioFeaturePreprocessor, VisualFeaturePreprocessor
from src.utils.feature_alignment import FeatureAligner
from src.utils.config_loader import load_telop_config

# pdはすでにインポート済み

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InferencePipeline:
    """動画編集推論パイプライン"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        fps: float = 10.0,
        num_tracks: int = 20,
        audio_preprocessor_path: str = None,
        visual_preprocessor_path: str = None,
        telop_config_path: str = None
    ):
        """
        初期化
        
        Args:
            model_path: 学習済みモデルのパス
            device: 'cpu' or 'cuda'
            fps: フレームレート（特徴量抽出用）
            num_tracks: トラック数
            audio_preprocessor_path: 音声前処理器のパス（Noneの場合は自動検出）
            visual_preprocessor_path: 映像前処理器のパス（Noneの場合は自動検出）
            telop_config_path: テロップ生成設定ファイルのパス（Noneの場合はデフォルト）
        """
        self.device = device
        
        # テロップ生成設定をロード
        self.telop_config = load_telop_config(telop_config_path)
        logger.info(f"Telop generation enabled: {self.telop_config.is_enabled()}")
        if self.telop_config.is_speech_enabled():
            logger.info("  Speech recognition: enabled")
        if self.telop_config.is_emotion_enabled():
            logger.info("  Emotion detection: enabled")
        self.fps = fps
        self.num_tracks = num_tracks
        
        # モデルをロード
        logger.info(f"Loading model from {model_path}")
        result = load_model(model_path, device=device)
        self.model = result['model']
        self.config = result['config']
        self.model.eval()
        
        # モデルタイプを判定（enable_multimodalまたはaudio_featuresの存在から）
        if 'model_type' not in self.config:
            if self.config.get('enable_multimodal', False) or ('audio_features' in self.config and 'visual_features' in self.config):
                self.config['model_type'] = 'multimodal'
            else:
                self.config['model_type'] = 'track_only'
        
        # 特徴量アライナー
        self.aligner = FeatureAligner(tolerance=0.05)
        
        # 前処理器をロード（推論時は正規化パラメータをロード）
        self.audio_preprocessor = None
        self.visual_preprocessor = None
        
        # マルチモーダルモデルの場合のみ前処理器をロード
        if self.config.get('model_type') == 'multimodal':
            # 自動検出: モデルと同じディレクトリから探す
            model_dir = Path(model_path).parent
            
            if audio_preprocessor_path is None:
                audio_preprocessor_path = model_dir / 'audio_preprocessor.pkl'
            if visual_preprocessor_path is None:
                visual_preprocessor_path = model_dir / 'visual_preprocessor.pkl'
            
            # 音声前処理器
            if Path(audio_preprocessor_path).exists():
                self.audio_preprocessor = AudioFeaturePreprocessor.load(str(audio_preprocessor_path))
                logger.info(f"  Loaded audio preprocessor from {audio_preprocessor_path}")
            else:
                logger.warning(f"  Audio preprocessor not found at {audio_preprocessor_path}")
                logger.warning("  Will use raw audio features (not recommended)")
            
            # 映像前処理器
            if Path(visual_preprocessor_path).exists():
                self.visual_preprocessor = VisualFeaturePreprocessor.load(str(visual_preprocessor_path))
                logger.info(f"  Loaded visual preprocessor from {visual_preprocessor_path}")
            else:
                logger.warning(f"  Visual preprocessor not found at {visual_preprocessor_path}")
                logger.warning("  Will use raw visual features (not recommended)")
        
        logger.info("Inference pipeline initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  FPS: {fps}")
        logger.info(f"  Num tracks: {num_tracks}")
        logger.info(f"  Model type: {self.config.get('model_type', 'track_only')}")
    
    def predict(
        self,
        video_path: str,
        output_xml_path: str,
        video_name: str = None
    ) -> str:
        """
        動画から編集を予測してXMLを生成
        
        Args:
            video_path: 入力動画のパス
            output_xml_path: 出力XMLのパス
            video_name: 動画名（Noneの場合はファイル名を使用）
        
        Returns:
            出力XMLファイルのパス
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting inference for: {video_path}")
        logger.info(f"{'='*80}\n")
        
        # 動画パスを保存（XML生成時に使用）
        self.video_path = video_path
        
        # 動画名を取得
        if video_name is None:
            video_name = Path(video_path).stem
        
        # ステップ1: 特徴量抽出
        logger.info("Step 1: Extracting features from video...")
        audio_features, visual_features = self._extract_features(video_path)
        
        # ステップ2: 特徴量の前処理とアライメント
        logger.info("Step 2: Preprocessing and aligning features...")
        aligned_features = self._preprocess_and_align(audio_features, visual_features)
        
        # ステップ3: モデルで予測
        logger.info("Step 3: Predicting edit parameters...")
        predictions = self._predict_with_model(aligned_features)
        
        # ステップ4: XMLに変換
        logger.info("Step 4: Converting predictions to Premiere Pro XML...")
        xml_path = self._create_xml(predictions, video_name, output_xml_path)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Inference complete!")
        logger.info(f"Output XML: {xml_path}")
        logger.info(f"{'='*80}\n")
        
        return xml_path
    
    def _extract_features(self, video_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        動画から特徴量を抽出
        
        Args:
            video_path: 動画ファイルのパス
        
        Returns:
            (audio_features, visual_features)
        """
        # 一時ディレクトリに特徴量を保存
        temp_dir = Path("temp_features")
        temp_dir.mkdir(exist_ok=True)
        
        video_name = Path(video_path).stem
        output_path = temp_dir / f"{video_name}_features.csv"
        
        # 特徴量抽出を実行（extract_video_features_parallel.pyのextract_features_worker関数を使用）
        logger.info(f"  特徴量抽出を開始...")
        
        # 既に抽出済みの場合はスキップ
        if output_path.exists():
            logger.info(f"  既存の特徴量ファイルを使用: {output_path}")
            df_all = pd.read_csv(output_path)
        else:
            # 特徴量抽出を実行
            logger.info(f"  動画を解析中...")
            from src.data_preparation.extract_video_features_parallel import extract_features_worker
            extract_features_worker(video_path, str(temp_dir))
            
            # 抽出されたCSVを読み込み
            if output_path.exists():
                df_all = pd.read_csv(output_path)
            else:
                raise FileNotFoundError(f"特徴量抽出に失敗しました: {output_path}")
        
        # 音声特徴量と映像特徴量に分割
        # 音声: time, audio_energy_rms, audio_is_speaking, silence_duration_ms, text_is_active, text_word, 
        #       telop_active, telop_text, speech_emb_0~5, telop_emb_0~5
        # 映像: scene_change, visual_motion, saliency_x, saliency_y, face_count, face_center_x, face_center_y, 
        #       face_size, face_mouth_open, face_eyebrow_raise, clip_0~clip_511
        
        # 基本音声特徴量
        audio_base_cols = ['time', 'audio_energy_rms', 'audio_is_speaking', 'silence_duration_ms', 'text_is_active']
        
        # テロップ関連
        if 'telop_active' in df_all.columns:
            audio_base_cols.append('telop_active')
        
        # テキスト埋め込み（音声認識）
        speech_emb_cols = [f'speech_emb_{i}' for i in range(6) if f'speech_emb_{i}' in df_all.columns]
        
        # テキスト埋め込み（テロップ）
        telop_emb_cols = [f'telop_emb_{i}' for i in range(6) if f'telop_emb_{i}' in df_all.columns]
        
        audio_cols = audio_base_cols + speech_emb_cols + telop_emb_cols
        
        # 映像特徴量
        visual_cols = ['time', 'scene_change', 'visual_motion', 'saliency_x', 'saliency_y', 
                      'face_count', 'face_center_x', 'face_center_y', 'face_size', 
                      'face_mouth_open', 'face_eyebrow_raise'] + [f'clip_{i}' for i in range(512)]
        
        # 存在するカラムのみを選択
        audio_cols = [col for col in audio_cols if col in df_all.columns]
        visual_cols = [col for col in visual_cols if col in df_all.columns]
        
        audio_df = df_all[audio_cols].copy()
        visual_df = df_all[visual_cols].copy()
        
        # テキスト埋め込みが不足している場合は追加
        if 'speech_emb_0' not in audio_df.columns or 'telop_emb_0' not in audio_df.columns:
            logger.info(f"  テキスト埋め込みを生成中...")
            from src.data_preparation.text_embedding import SimpleTextEmbedder
            embedder = SimpleTextEmbedder()
            
            # 音声認識のテキスト埋め込み
            if 'text_word' in audio_df.columns:
                speech_embeddings = audio_df['text_word'].apply(
                    lambda x: embedder.embed(x if pd.notna(x) else "")
                )
                for i in range(6):
                    audio_df[f'speech_emb_{i}'] = speech_embeddings.apply(lambda e: e[i])
            else:
                for i in range(6):
                    audio_df[f'speech_emb_{i}'] = 0.0
            
            # テロップのテキスト埋め込み
            if 'telop_text' in audio_df.columns:
                telop_embeddings = audio_df['telop_text'].apply(
                    lambda x: embedder.embed(x if pd.notna(x) else "")
                )
                for i in range(6):
                    audio_df[f'telop_emb_{i}'] = telop_embeddings.apply(lambda e: e[i])
            else:
                for i in range(6):
                    audio_df[f'telop_emb_{i}'] = 0.0
        
        logger.info(f"  音声特徴量: {len(audio_df)} timesteps, {audio_df.shape[1]-1} features")
        logger.info(f"  映像特徴量: {len(visual_df)} timesteps, {len(visual_cols)-1} features")
        
        return audio_df, visual_df
    
    def _preprocess_and_align(
        self,
        audio_df: pd.DataFrame,
        visual_df: pd.DataFrame
    ) -> Dict[str, torch.Tensor]:
        """
        特徴量を前処理してアライメント
        
        Args:
            audio_df: 音声特徴量
            visual_df: 映像特徴量
        
        Returns:
            アライメント済み特徴量の辞書
        """
        # タイムスタンプを生成（FPSベース）
        max_time = max(
            audio_df['time'].max() if len(audio_df) > 0 else 0,
            visual_df['time'].max() if len(visual_df) > 0 else 0
        )
        num_timesteps = int(max_time * self.fps) + 1
        track_times = np.arange(num_timesteps) / self.fps
        
        logger.info(f"  目標タイムステップ数: {num_timesteps}")
        logger.info(f"  動画の長さ: {max_time:.2f}秒")
        
        # アライメント
        aligned_audio, aligned_visual, modality_mask, stats = self.aligner.align_features(
            track_times, audio_df, visual_df, video_id="inference"
        )
        
        logger.info(f"  アライメント統計:")
        logger.info(f"    音声カバレッジ: {stats.get('audio_coverage_pct', 0):.1f}%")
        logger.info(f"    映像カバレッジ: {stats.get('visual_coverage_pct', 0):.1f}%")
        
        # 前処理（正規化）を適用
        # 注意: 前処理器が古い次元（4次元）用の場合はスキップ
        if aligned_audio is not None and self.audio_preprocessor is not None:
            try:
                logger.info(f"  音声特徴量を正規化中...")
                aligned_audio = self.audio_preprocessor.transform(aligned_audio)
            except ValueError as e:
                logger.warning(f"  音声前処理をスキップ（次元不一致）: {e}")
                logger.warning(f"  正規化なしで続行します")
        
        if aligned_visual is not None and self.visual_preprocessor is not None:
            try:
                logger.info(f"  映像特徴量を正規化中...")
                # face_countを取得（正規化に必要）
                face_counts = visual_df['face_count'].values if 'face_count' in visual_df.columns else None
                aligned_visual = self.visual_preprocessor.transform(aligned_visual, face_counts)
            except ValueError as e:
                logger.warning(f"  映像前処理をスキップ（次元不一致）: {e}")
                logger.warning(f"  正規化なしで続行します")
        
        # Tensorに変換
        features = {}
        
        # 音声特徴量の次元を取得（設定から）
        audio_dim = self.config.get('audio_features', 17)  # デフォルト17次元（テロップ対応）
        
        if aligned_audio is not None:
            # NaN/Infを0に置換（重要: モデルがNaNを処理できないため）
            aligned_audio = np.nan_to_num(aligned_audio, nan=0.0, posinf=0.0, neginf=0.0)
            features['audio'] = torch.from_numpy(aligned_audio).float().unsqueeze(0)  # (1, seq_len, audio_dim)
            logger.info(f"  音声特徴量の形状: {features['audio'].shape}")
        else:
            features['audio'] = torch.zeros(1, num_timesteps, audio_dim)
            logger.info(f"  音声特徴量なし（ゼロパディング）: {features['audio'].shape}")
        
        # 視覚特徴量の次元を取得（設定から）
        visual_dim = self.config.get('visual_features', 522)  # デフォルト522次元
        
        if aligned_visual is not None:
            # NaN/Infを0に置換（重要: モデルがNaNを処理できないため）
            aligned_visual = np.nan_to_num(aligned_visual, nan=0.0, posinf=0.0, neginf=0.0)
            features['visual'] = torch.from_numpy(aligned_visual).float().unsqueeze(0)  # (1, seq_len, visual_dim)
            logger.info(f"  映像特徴量の形状: {features['visual'].shape}")
        else:
            features['visual'] = torch.zeros(1, num_timesteps, visual_dim)
            logger.info(f"  映像特徴量なし（ゼロパディング）: {features['visual'].shape}")
        
        # ダミーのトラック特徴量（推論時は不要だが、モデル入力として必要）
        # モデルの入力次元に合わせる
        track_features_dim = self.config.get('input_features', self.config.get('track_features', 240))
        features['track'] = torch.zeros(1, num_timesteps, track_features_dim)
        
        # モダリティマスク
        features['modality_mask'] = torch.from_numpy(modality_mask).bool().unsqueeze(0)  # (1, seq_len, 3)
        
        # パディングマスク（全てTrue = パディングなし）
        features['padding_mask'] = torch.ones(1, num_timesteps, dtype=torch.bool)
        
        return features
    
    def _predict_with_model(self, features: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        モデルで編集パラメータを予測
        
        Args:
            features: 入力特徴量
        
        Returns:
            予測結果の辞書
        """
        # シーケンス長を取得
        seq_len = features['track'].shape[1]
        max_chunk_size = 5000  # 位置エンコーディングの最大長
        
        # チャンク分割が必要かチェック
        if seq_len > max_chunk_size:
            logger.info(f"  シーケンス長 {seq_len} > {max_chunk_size}、チャンク分割して処理します")
            return self._predict_with_chunks(features, max_chunk_size)
        
        # デバイスに転送
        for key in features:
            features[key] = features[key].to(self.device)
        
        # モデルタイプに応じて推論
        model_type = self.config.get('model_type', 'track_only')
        
        with torch.no_grad():
            if model_type == 'multimodal':
                # マルチモーダルモデル
                outputs = self.model(
                    audio=features['audio'],
                    visual=features['visual'],
                    track=features['track'],
                    padding_mask=features['padding_mask'],
                    modality_mask=features['modality_mask']
                )
            else:
                # トラックオンリーモデル（古いモデル）
                logger.info(f"  ⚠️  トラックオンリーモデルを使用（音声・映像特徴量は無視されます）")
                # トラック特徴量のみを使用（ダミーのゼロ値）
                outputs = self.model(
                    features['track'],
                    features['padding_mask']
                )
        
        # CPUに戻してnumpy配列に変換
        predictions = {}
        for key, value in outputs.items():
            predictions[key] = value.cpu().numpy()[0]  # (seq_len, ...)
            # NaN/Infチェック
            if np.isnan(predictions[key]).any():
                logger.warning(f"  ⚠️  {key} contains NaN values!")
            if np.isinf(predictions[key]).any():
                logger.warning(f"  ⚠️  {key} contains Inf values!")
    
    def _predict_with_chunks(self, features: Dict[str, torch.Tensor], chunk_size: int) -> Dict[str, np.ndarray]:
        """
        長いシーケンスをチャンクに分割して予測
        
        Args:
            features: 入力特徴量
            chunk_size: チャンクサイズ
        
        Returns:
            予測結果の辞書
        """
        seq_len = features['track'].shape[1]
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        logger.info(f"  {num_chunks}個のチャンクに分割")
        
        # 結果を格納するリスト
        all_predictions = {}
        
        model_type = self.config.get('model_type', 'track_only')
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, seq_len)
            logger.info(f"  チャンク {i+1}/{num_chunks}: {start_idx}-{end_idx}")
            
            # チャンクを抽出
            chunk_features = {}
            for key, value in features.items():
                chunk_features[key] = value[:, start_idx:end_idx].to(self.device)
            
            # 推論
            with torch.no_grad():
                if model_type == 'multimodal':
                    outputs = self.model(
                        audio=chunk_features['audio'],
                        visual=chunk_features['visual'],
                        track=chunk_features['track'],
                        padding_mask=chunk_features['padding_mask'],
                        modality_mask=chunk_features['modality_mask']
                    )
                else:
                    outputs = self.model(
                        chunk_features['track'],
                        chunk_features['padding_mask']
                    )
            
            # 結果を保存
            for key, value in outputs.items():
                if key not in all_predictions:
                    all_predictions[key] = []
                all_predictions[key].append(value.cpu().numpy()[0])
        
        # チャンクを結合
        predictions = {}
        for key, chunks in all_predictions.items():
            predictions[key] = np.concatenate(chunks, axis=0)
            logger.info(f"  {key}: {predictions[key].shape}")
        
        return predictions
        
        logger.info(f"  予測完了: {len(predictions)}種類のパラメータ")
        logger.info(f"  出力キー: {list(predictions.keys())}")
        logger.info(f"  シーケンス長: {list(predictions.values())[0].shape[0]}")
        
        return predictions
    
    def _extract_telop_info(self, video_name: str) -> list:
        """
        抽出済みのテロップ情報を取得
        
        Args:
            video_name: 動画名
        
        Returns:
            テロップ情報のリスト
        """
        # temp_featuresから特徴量CSVを読み込み
        csv_path = Path("temp_features") / f"{video_name}_features.csv"
        if not csv_path.exists():
            return []
        
        try:
            df = pd.read_csv(csv_path)
            
            # テロップがアクティブな時間帯を抽出
            telops = []
            if 'telop_active' in df.columns and 'telop_text' in df.columns:
                # テロップがアクティブな区間を検出
                active_mask = df['telop_active'] == 1
                if active_mask.any():
                    # 連続する区間をグループ化
                    changes = active_mask.astype(int).diff().fillna(0)
                    starts = df[changes == 1].index.tolist()
                    ends = df[changes == -1].index.tolist()
                    
                    # 最初から始まる場合
                    if active_mask.iloc[0]:
                        starts = [0] + starts
                    
                    # 最後まで続く場合
                    if active_mask.iloc[-1]:
                        ends = ends + [len(df) - 1]
                    
                    # 各区間のテロップを抽出
                    for start_idx, end_idx in zip(starts, ends):
                        # その区間の最初の非NaNテキストを取得
                        segment_texts = df.loc[start_idx:end_idx, 'telop_text'].dropna()
                        if len(segment_texts) > 0:
                            text = segment_texts.iloc[0]
                            start_time = df.loc[start_idx, 'time']
                            end_time = df.loc[end_idx, 'time']
                            
                            telops.append({
                                'text': text,
                                'start_frame': int(start_time * self.fps),
                                'end_frame': int(end_time * self.fps)
                            })
            
            logger.info(f"  Extracted {len(telops)} telop segments")
            return telops
            
        except Exception as e:
            logger.warning(f"  Failed to extract telop info: {e}")
            return []
    
    def _extract_speech_telops(self, video_path: str, video_name: str) -> list:
        """
        Extract speech and generate telops using ASR
        
        Args:
            video_path: Path to video file
            video_name: Video name for caching
        
        Returns:
            List of dicts with keys: 'text', 'start_frame', 'end_frame', 'type'='speech'
        """
        if not self.telop_config.is_speech_enabled():
            logger.info("  Speech recognition disabled")
            return []
        
        try:
            # Check for cached results
            cache_dir = Path("temp_features")
            cache_dir.mkdir(exist_ok=True)
            cache_path = cache_dir / f"{video_name}_speech.json"
            
            speech_config = self.telop_config.get_speech_config()
            
            if cache_path.exists() and speech_config.get('cache_results', True):
                logger.info(f"  Loading cached speech recognition results from {cache_path}")
                import json
                with open(cache_path, 'r', encoding='utf-8') as f:
                    segments = json.load(f)
                logger.info(f"  Loaded {len(segments)} speech segments from cache")
                return segments
            
            # Import Whisper
            try:
                import whisper
            except ImportError:
                logger.warning("  Whisper not installed. Install with: pip install openai-whisper")
                logger.warning("  Skipping speech recognition")
                return []
            
            # Load Whisper model
            model_size = speech_config.get('model_size', 'small')
            language = speech_config.get('language', 'ja')
            
            logger.info(f"  Loading Whisper model: {model_size}")
            model = whisper.load_model(model_size)
            
            # Transcribe audio
            logger.info(f"  Transcribing audio (language: {language})...")
            result = whisper.transcribe(
                model,
                video_path,
                language=language,
                verbose=False
            )
            
            # Extract segments
            segments = []
            for segment in result['segments']:
                text = segment['text'].strip()
                if not text:
                    continue
                
                start_time = segment['start']
                end_time = segment['end']
                
                segments.append({
                    'text': text,
                    'start_frame': int(start_time * self.fps),
                    'end_frame': int(end_time * self.fps),
                    'type': 'speech',
                    'start_time': start_time,
                    'end_time': end_time
                })
            
            logger.info(f"  Extracted {len(segments)} raw speech segments")
            
            # Process segments (merge short, split long)
            segments = self._process_speech_segments(segments, speech_config)
            logger.info(f"  Processed to {len(segments)} speech segments")
            
            # Cache results
            if speech_config.get('cache_results', True):
                import json
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(segments, f, ensure_ascii=False, indent=2)
                logger.info(f"  Cached speech recognition results to {cache_path}")
            
            return segments
            
        except Exception as e:
            logger.error(f"  Speech recognition failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _process_speech_segments(self, segments: list, config: dict) -> list:
        """
        Process speech segments: merge short segments, split long segments
        
        Args:
            segments: List of speech segments
            config: Speech configuration
        
        Returns:
            Processed list of segments
        """
        if not segments:
            return segments
        
        min_duration = config.get('min_segment_duration', 0.5)
        max_duration = config.get('max_segment_duration', 5.0)
        
        # Step 1: Merge short segments
        merged_segments = []
        i = 0
        while i < len(segments):
            current = segments[i]
            duration = current['end_time'] - current['start_time']
            
            # If segment is short and there's a next segment, try to merge
            if duration < min_duration and i + 1 < len(segments):
                next_seg = segments[i + 1]
                gap = next_seg['start_time'] - current['end_time']
                
                # Merge if gap is small (< 0.5 seconds)
                if gap < 0.5:
                    merged = {
                        'text': current['text'] + ' ' + next_seg['text'],
                        'start_time': current['start_time'],
                        'end_time': next_seg['end_time'],
                        'start_frame': current['start_frame'],
                        'end_frame': next_seg['end_frame'],
                        'type': 'speech'
                    }
                    merged_segments.append(merged)
                    i += 2  # Skip next segment
                    continue
            
            merged_segments.append(current)
            i += 1
        
        # Step 2: Split long segments
        final_segments = []
        for segment in merged_segments:
            duration = segment['end_time'] - segment['start_time']
            
            if duration > max_duration:
                # Split at sentence boundaries (。、！？)
                text = segment['text']
                sentences = []
                current_sentence = ""
                
                for char in text:
                    current_sentence += char
                    if char in ['。', '、', '！', '？', '.', ',', '!', '?']:
                        sentences.append(current_sentence.strip())
                        current_sentence = ""
                
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                
                # If we have multiple sentences, split proportionally
                if len(sentences) > 1:
                    time_per_char = duration / len(text)
                    current_time = segment['start_time']
                    
                    for sentence in sentences:
                        sentence_duration = len(sentence) * time_per_char
                        end_time = min(current_time + sentence_duration, segment['end_time'])
                        
                        final_segments.append({
                            'text': sentence,
                            'start_time': current_time,
                            'end_time': end_time,
                            'start_frame': int(current_time * self.fps),
                            'end_frame': int(end_time * self.fps),
                            'type': 'speech'
                        })
                        
                        current_time = end_time
                else:
                    # Can't split, keep as is
                    final_segments.append(segment)
            else:
                final_segments.append(segment)
        
        return final_segments
    
    def _detect_emotion_telops(self, video_path: str, video_name: str) -> list:
        """
        Detect emotions and generate telops
        
        Args:
            video_path: Path to video file
            video_name: Video name for caching
        
        Returns:
            List of dicts with keys: 'text', 'start_frame', 'end_frame', 'type'='emotion', 'emotion_type'
        """
        if not self.telop_config.is_emotion_enabled():
            logger.info("  Emotion detection disabled")
            return []
        
        try:
            # Check for cached results
            cache_dir = Path("temp_features")
            cache_dir.mkdir(exist_ok=True)
            cache_path = cache_dir / f"{video_name}_emotions.json"
            
            emotion_config = self.telop_config.get_emotion_config()
            
            if cache_path.exists():
                logger.info(f"  Loading cached emotion detection results from {cache_path}")
                import json
                with open(cache_path, 'r', encoding='utf-8') as f:
                    emotions = json.load(f)
                logger.info(f"  Loaded {len(emotions)} emotion segments from cache")
                return emotions
            
            # Import audio processing libraries
            try:
                import librosa
                import soundfile as sf
            except ImportError:
                logger.warning("  librosa or soundfile not installed. Install with: pip install librosa soundfile")
                logger.warning("  Skipping emotion detection")
                return []
            
            # Load audio
            logger.info(f"  Loading audio for emotion detection...")
            y, sr = librosa.load(video_path, sr=22050)
            
            # Extract audio features
            logger.info(f"  Extracting audio features...")
            features = self._extract_audio_features_for_emotion(y, sr)
            
            # Detect emotions
            logger.info(f"  Detecting emotions...")
            emotions = []
            
            # Detect laughter
            if emotion_config.get('laughter', {}).get('enabled', True):
                laughter_segments = self._detect_laughter(features, emotion_config)
                emotions.extend(laughter_segments)
            
            # Detect surprise
            if emotion_config.get('surprise', {}).get('enabled', True):
                surprise_segments = self._detect_surprise(features, emotion_config)
                emotions.extend(surprise_segments)
            
            # Detect sadness
            if emotion_config.get('sadness', {}).get('enabled', True):
                sadness_segments = self._detect_sadness(features, emotion_config)
                emotions.extend(sadness_segments)
            
            # Sort by start time
            emotions.sort(key=lambda x: x['start_time'])
            
            logger.info(f"  Detected {len(emotions)} emotion segments")
            
            # Cache results
            import json
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(emotions, f, ensure_ascii=False, indent=2)
            logger.info(f"  Cached emotion detection results to {cache_path}")
            
            return emotions
            
        except Exception as e:
            logger.error(f"  Emotion detection failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _extract_audio_features_for_emotion(self, y: np.ndarray, sr: int) -> dict:
        """
        Extract audio features for emotion detection
        
        Args:
            y: Audio signal
            sr: Sample rate
        
        Returns:
            Dictionary of features
        """
        import librosa
        
        # Frame-level features (hop_length = 512 samples ≈ 0.023s at 22050Hz)
        hop_length = 512
        
        # Energy (RMS)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        
        # Pitch (F0) using piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)
        # Get the pitch with highest magnitude at each frame
        pitch = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch.append(pitches[index, t])
        pitch = np.array(pitch)
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        
        # Time axis
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        return {
            'times': times,
            'rms': rms,
            'zcr': zcr,
            'pitch': pitch,
            'mfccs': mfccs,
            'sr': sr,
            'hop_length': hop_length
        }
    
    def _detect_laughter(self, features: dict, config: dict) -> list:
        """
        Detect laughter segments
        
        Args:
            features: Audio features
            config: Emotion configuration
        
        Returns:
            List of laughter segments
        """
        laughter_config = config.get('laughter', {})
        pitch_std_threshold = laughter_config.get('pitch_std_threshold', 50.0)
        energy_threshold = laughter_config.get('energy_threshold', 0.3)
        confidence_threshold = config.get('confidence_threshold', 0.6)
        
        times = features['times']
        pitch = features['pitch']
        rms = features['rms']
        
        # Normalize RMS
        rms_normalized = rms / (rms.max() + 1e-8)
        
        # Detect laughter: high pitch variation + high energy
        window_size = 20  # ~0.5 seconds
        laughter_segments = []
        
        for i in range(0, len(times) - window_size, window_size // 2):
            window_pitch = pitch[i:i+window_size]
            window_rms = rms_normalized[i:i+window_size]
            
            # Filter out zero pitches
            valid_pitch = window_pitch[window_pitch > 0]
            if len(valid_pitch) < window_size // 2:
                continue
            
            pitch_std = np.std(valid_pitch)
            mean_energy = np.mean(window_rms)
            
            # Check thresholds
            if pitch_std > pitch_std_threshold and mean_energy > energy_threshold:
                confidence = min(1.0, (pitch_std / pitch_std_threshold + mean_energy / energy_threshold) / 2)
                
                if confidence >= confidence_threshold:
                    start_time = times[i]
                    end_time = times[min(i + window_size, len(times) - 1)]
                    duration = end_time - start_time
                    
                    # Choose text based on duration
                    if duration < 1.0:
                        text = laughter_config.get('text_short', 'w')
                    elif duration < 2.0:
                        text = laughter_config.get('text_medium', 'www')
                    else:
                        text = laughter_config.get('text_long', 'wwww')
                    
                    laughter_segments.append({
                        'text': text,
                        'start_time': start_time,
                        'end_time': end_time,
                        'start_frame': int(start_time * self.fps),
                        'end_frame': int(end_time * self.fps),
                        'type': 'emotion',
                        'emotion_type': 'laughter',
                        'confidence': float(confidence)
                    })
        
        return laughter_segments
    
    def _detect_surprise(self, features: dict, config: dict) -> list:
        """
        Detect surprise segments
        
        Args:
            features: Audio features
            config: Emotion configuration
        
        Returns:
            List of surprise segments
        """
        surprise_config = config.get('surprise', {})
        pitch_delta_threshold = surprise_config.get('pitch_delta_threshold', 100.0)
        max_duration = surprise_config.get('max_duration', 1.0)
        confidence_threshold = config.get('confidence_threshold', 0.6)
        
        times = features['times']
        pitch = features['pitch']
        
        # Detect surprise: sudden pitch increase
        surprise_segments = []
        
        for i in range(1, len(pitch)):
            if pitch[i] > 0 and pitch[i-1] > 0:
                pitch_delta = pitch[i] - pitch[i-1]
                
                if pitch_delta > pitch_delta_threshold:
                    confidence = min(1.0, pitch_delta / pitch_delta_threshold)
                    
                    if confidence >= confidence_threshold:
                        start_time = times[i-1]
                        end_time = min(times[i] + max_duration, times[-1])
                        
                        text = surprise_config.get('text', '！')
                        
                        surprise_segments.append({
                            'text': text,
                            'start_time': start_time,
                            'end_time': end_time,
                            'start_frame': int(start_time * self.fps),
                            'end_frame': int(end_time * self.fps),
                            'type': 'emotion',
                            'emotion_type': 'surprise',
                            'confidence': float(confidence)
                        })
        
        return surprise_segments
    
    def _detect_sadness(self, features: dict, config: dict) -> list:
        """
        Detect sadness segments
        
        Args:
            features: Audio features
            config: Emotion configuration
        
        Returns:
            List of sadness segments
        """
        sadness_config = config.get('sadness', {})
        pitch_mean_threshold = sadness_config.get('pitch_mean_threshold', 150.0)
        energy_threshold = sadness_config.get('energy_threshold', 0.1)
        confidence_threshold = config.get('confidence_threshold', 0.6)
        
        times = features['times']
        pitch = features['pitch']
        rms = features['rms']
        
        # Normalize RMS
        rms_normalized = rms / (rms.max() + 1e-8)
        
        # Detect sadness: low pitch + low energy
        window_size = 40  # ~1 second
        sadness_segments = []
        
        for i in range(0, len(times) - window_size, window_size // 2):
            window_pitch = pitch[i:i+window_size]
            window_rms = rms_normalized[i:i+window_size]
            
            # Filter out zero pitches
            valid_pitch = window_pitch[window_pitch > 0]
            if len(valid_pitch) < window_size // 2:
                continue
            
            pitch_mean = np.mean(valid_pitch)
            mean_energy = np.mean(window_rms)
            
            # Check thresholds (inverted for sadness)
            if pitch_mean < pitch_mean_threshold and mean_energy < energy_threshold:
                confidence = min(1.0, (1 - pitch_mean / pitch_mean_threshold + 1 - mean_energy / energy_threshold) / 2)
                
                if confidence >= confidence_threshold:
                    start_time = times[i]
                    end_time = times[min(i + window_size, len(times) - 1)]
                    
                    text = sadness_config.get('text', '...')
                    
                    sadness_segments.append({
                        'text': text,
                        'start_time': start_time,
                        'end_time': end_time,
                        'start_frame': int(start_time * self.fps),
                        'end_frame': int(end_time * self.fps),
                        'type': 'emotion',
                        'emotion_type': 'sadness',
                        'confidence': float(confidence)
                    })
        
        return sadness_segments
    
    def _create_xml(
        self,
        predictions: Dict[str, np.ndarray],
        video_name: str,
        output_path: str
    ) -> str:
        """
        予測結果からPremiere Pro XMLを生成
        
        Args:
            predictions: モデルの予測結果
            video_name: 動画名
            output_path: 出力XMLパス
        
        Returns:
            出力XMLファイルのパス
        """
        # 予測結果を解釈
        seq_len = predictions['active'].shape[0]
        
        # Active logitsをprobabilitiesに変換
        # predictions['active']の形状: (seq_len, num_tracks, 2) - [inactive_logit, active_logit]
        # Softmaxを適用して確率に変換し、active class (index 1) の確率を取得
        import scipy.special
        active_probs = scipy.special.softmax(predictions['active'], axis=-1)[:, :, 1]  # (seq_len, num_tracks)
        
        # 各フレームで最もアクティブなトラックを選択（重複を防ぐ）
        # 閾値0.29を超えるトラックの中から、最も確率が高いものを選択
        best_track_per_frame = np.full(seq_len, -1, dtype=int)  # -1 = どのトラックもアクティブでない
        
        for t in range(seq_len):
            # このフレームで閾値を超えるトラックを探す
            active_tracks = np.where(active_probs[t, :] > 0.29)[0]
            if len(active_tracks) > 0:
                # 最も確率が高いトラックを選択
                best_track = active_tracks[np.argmax(active_probs[t, active_tracks])]
                best_track_per_frame[t] = best_track
        
        # 連続する同じトラックの区間をグループ化
        tracks_data = []
        if np.any(best_track_per_frame >= 0):
            current_track = best_track_per_frame[0]
            start_frame = 0
            
            for t in range(1, seq_len):
                if best_track_per_frame[t] != current_track:
                    # トラックが変わった、または非アクティブになった
                    if current_track >= 0:
                        # 前のトラックのセグメントを保存
                        end_frame = t - 1
                        
                        # このセグメントのパラメータを計算
                        track_data = {
                            'track_id': current_track,
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'asset_id': int(np.median(predictions['asset'][start_frame:end_frame+1, current_track].argmax(axis=-1))) if 'asset' in predictions else 0,
                            'scale': float(np.mean(predictions['scale'][start_frame:end_frame+1, current_track])),
                            'position_x': float(np.mean(predictions['pos_x'][start_frame:end_frame+1, current_track])) if 'pos_x' in predictions else 0.0,
                            'position_y': float(np.mean(predictions['pos_y'][start_frame:end_frame+1, current_track])) if 'pos_y' in predictions else 0.0,
                            'crop_left': float(np.mean(predictions['crop_l'][start_frame:end_frame+1, current_track])) if 'crop_l' in predictions else 0.0,
                            'crop_right': float(np.mean(predictions['crop_r'][start_frame:end_frame+1, current_track])) if 'crop_r' in predictions else 0.0,
                            'crop_top': float(np.mean(predictions['crop_t'][start_frame:end_frame+1, current_track])) if 'crop_t' in predictions else 0.0,
                            'crop_bottom': float(np.mean(predictions['crop_b'][start_frame:end_frame+1, current_track])) if 'crop_b' in predictions else 0.0,
                        }
                        tracks_data.append(track_data)
                    
                    # 新しいセグメントを開始
                    current_track = best_track_per_frame[t]
                    start_frame = t
            
            # 最後のセグメントを保存
            if current_track >= 0:
                end_frame = seq_len - 1
                track_data = {
                    'track_id': current_track,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'asset_id': int(np.median(predictions['asset'][start_frame:end_frame+1, current_track].argmax(axis=-1))) if 'asset' in predictions else 0,
                    'scale': float(np.mean(predictions['scale'][start_frame:end_frame+1, current_track])),
                    'position_x': float(np.mean(predictions['pos_x'][start_frame:end_frame+1, current_track])) if 'pos_x' in predictions else 0.0,
                    'position_y': float(np.mean(predictions['pos_y'][start_frame:end_frame+1, current_track])) if 'pos_y' in predictions else 0.0,
                    'crop_left': float(np.mean(predictions['crop_l'][start_frame:end_frame+1, current_track])) if 'crop_l' in predictions else 0.0,
                    'crop_right': float(np.mean(predictions['crop_r'][start_frame:end_frame+1, current_track])) if 'crop_r' in predictions else 0.0,
                    'crop_top': float(np.mean(predictions['crop_t'][start_frame:end_frame+1, current_track])) if 'crop_t' in predictions else 0.0,
                    'crop_bottom': float(np.mean(predictions['crop_b'][start_frame:end_frame+1, current_track])) if 'crop_b' in predictions else 0.0,
                }
                tracks_data.append(track_data)
        
        # デバッグ: active予測の統計を表示
        active_logits = predictions['active']  # (seq_len, num_tracks, 2)
        
        # Convert logits to probabilities
        import scipy.special
        active_probs = scipy.special.softmax(active_logits, axis=-1)[:, :, 1]  # (seq_len, num_tracks)
        
        logger.info(f"  Active prediction stats (logits):")
        logger.info(f"    Min: {active_logits.min():.4f}, Max: {active_logits.max():.4f}, Mean: {active_logits.mean():.4f}")
        
        logger.info(f"  Active prediction stats (probabilities after softmax):")
        logger.info(f"    Min: {active_probs.min():.4f}, Max: {active_probs.max():.4f}, Mean: {active_probs.mean():.4f}")
        logger.info(f"    Median: {np.median(active_probs):.4f}, Std: {active_probs.std():.4f}")
        logger.info(f"    Percentiles: 25%={np.percentile(active_probs, 25):.4f}, 50%={np.percentile(active_probs, 50):.4f}, 75%={np.percentile(active_probs, 75):.4f}, 90%={np.percentile(active_probs, 90):.4f}")
        logger.info(f"    Values > 0.5: {(active_probs > 0.5).sum()} / {active_probs.size}")
        logger.info(f"    Values > 0.3: {(active_probs > 0.3).sum()} / {active_probs.size}")
        logger.info(f"    Values > 0.1: {(active_probs > 0.1).sum()} / {active_probs.size}")
        
        logger.info(f"  Detected {len(tracks_data)} active track segments")
        
        # テロップ情報を抽出（OCR）
        telops = self._extract_telop_info(video_name)
        
        # AI字幕を生成
        logger.info("Step 4.1: Generating AI telops...")
        ai_telops = []
        
        # 音声認識による字幕
        if self.telop_config.is_speech_enabled():
            logger.info("  Extracting speech telops...")
            speech_telops = self._extract_speech_telops(self.video_path, video_name)
            ai_telops.extend(speech_telops)
            logger.info(f"  Added {len(speech_telops)} speech telops")
        
        # 感情検出による字幕
        if self.telop_config.is_emotion_enabled():
            logger.info("  Detecting emotion telops...")
            emotion_telops = self._detect_emotion_telops(self.video_path, video_name)
            ai_telops.extend(emotion_telops)
            logger.info(f"  Added {len(emotion_telops)} emotion telops")
        
        logger.info(f"  Total AI telops: {len(ai_telops)}")
        
        # OTIOを使ってXMLを生成（音声クリップはXMLレベルで修正）
        from src.inference.otio_xml_generator import create_premiere_xml_with_otio
        
        xml_path = create_premiere_xml_with_otio(
            video_path=self.video_path,
            video_name=video_name,
            total_frames=seq_len,
            fps=self.fps,
            tracks_data=tracks_data,
            telops=telops,
            ai_telops=ai_telops,
            output_path=output_path
        )
        
        return xml_path
    
    def _generate_xml_content(
        self,
        tracks_data: list,
        video_name: str,
        total_frames: int,
        fps: float,
        video_path: str = None,
        telops: list = None
    ) -> str:
        """
        Premiere Pro XML形式のコンテンツを生成（音声トラックとテロップ対応）
        
        Args:
            tracks_data: トラックデータのリスト
            video_name: 動画名
            total_frames: 総フレーム数
            fps: フレームレート
            video_path: 元動画のパス（音声トラック用）
            telops: テロップ情報のリスト
        
        Returns:
            XML文字列
        """
        if telops is None:
            telops = []
        import xml.etree.ElementTree as ET
        from xml.dom import minidom
        
        # ルート要素を作成
        root = ET.Element('xmeml', version='5')
        
        # シーケンスを作成
        sequence = ET.SubElement(root, 'sequence', id=video_name)
        ET.SubElement(sequence, 'name').text = video_name
        ET.SubElement(sequence, 'duration').text = str(total_frames)
        
        # レート設定
        rate = ET.SubElement(sequence, 'rate')
        ET.SubElement(rate, 'timebase').text = str(int(fps))
        ET.SubElement(rate, 'ntsc').text = 'FALSE'
        
        # メディア
        media = ET.SubElement(sequence, 'media')
        
        # ビデオトラック
        video = ET.SubElement(media, 'video')
        
        # フォーマット設定（縦長動画対応）
        fmt = ET.SubElement(video, 'format')
        sc = ET.SubElement(fmt, 'samplecharacteristics')
        ET.SubElement(sc, 'width').text = '1080'
        ET.SubElement(sc, 'height').text = '1920'
        ET.SubElement(sc, 'pixelaspectratio').text = 'square'
        
        # ビデオトラックを追加
        for track_data in tracks_data:
            track = ET.SubElement(video, 'track')
            
            # クリップアイテム
            clipitem = ET.SubElement(track, 'clipitem', id=f"clip_{track_data['track_id']}")
            ET.SubElement(clipitem, 'name').text = video_name
            ET.SubElement(clipitem, 'start').text = str(track_data['start_frame'])
            ET.SubElement(clipitem, 'end').text = str(track_data['end_frame'])
            ET.SubElement(clipitem, 'in').text = str(track_data['start_frame'])
            ET.SubElement(clipitem, 'out').text = str(track_data['end_frame'])
            
            # ファイル参照（共通のファイルIDを使用）
            file_elem = ET.SubElement(clipitem, 'file', id="file_main")
            ET.SubElement(file_elem, 'name').text = video_name
            if video_path:
                # パスをURLエンコード
                from urllib.parse import quote
                encoded_path = quote(video_path.replace('\\', '/'))
                ET.SubElement(file_elem, 'pathurl').text = f"file://localhost/{encoded_path}"
            
            # モーションエフェクト
            self._add_motion_effect(clipitem, track_data['scale'], 
                                   track_data['position_x'], track_data['position_y'])
            
            # クロップエフェクト
            self._add_crop_effect(clipitem, track_data['crop_left'], track_data['crop_right'],
                                 track_data['crop_top'], track_data['crop_bottom'])
        
        # 音声トラックを追加
        audio = ET.SubElement(media, 'audio')
        
        # 音声フォーマット
        audio_fmt = ET.SubElement(audio, 'format')
        audio_sc = ET.SubElement(audio_fmt, 'samplecharacteristics')
        ET.SubElement(audio_sc, 'depth').text = '16'
        ET.SubElement(audio_sc, 'samplerate').text = '48000'
        
        # 音声トラック（ステレオなので2トラック）
        for channel_idx in range(2):
            audio_track = ET.SubElement(audio, 'track')
            
            # 音声クリップアイテム
            audio_clipitem = ET.SubElement(audio_track, 'clipitem', id=f"audioclip_{channel_idx}")
            ET.SubElement(audio_clipitem, 'name').text = video_name
            ET.SubElement(audio_clipitem, 'enabled').text = 'TRUE'
            ET.SubElement(audio_clipitem, 'duration').text = str(total_frames)
            ET.SubElement(audio_clipitem, 'start').text = '0'
            ET.SubElement(audio_clipitem, 'end').text = str(total_frames)
            ET.SubElement(audio_clipitem, 'in').text = '0'
            ET.SubElement(audio_clipitem, 'out').text = str(total_frames)
            
            # 同じファイルを参照
            ET.SubElement(audio_clipitem, 'file', id="file_main")
            
            # ソーストラック指定
            sourcetrack = ET.SubElement(audio_clipitem, 'sourcetrack')
            ET.SubElement(sourcetrack, 'mediatype').text = 'audio'
            ET.SubElement(sourcetrack, 'trackindex').text = str(channel_idx + 1)
        
        # テロップを追加（ビデオトラックの最後に）
        if telops:
            logger.info(f"  Adding {len(telops)} telop effects to XML")
            for telop_idx, telop in enumerate(telops):
                telop_track = ET.SubElement(video, 'track')
                
                # テロップクリップアイテム
                telop_clipitem = ET.SubElement(telop_track, 'clipitem', id=f"telop_{telop_idx}")
                ET.SubElement(telop_clipitem, 'name').text = 'グラフィック'
                ET.SubElement(telop_clipitem, 'enabled').text = 'TRUE'
                ET.SubElement(telop_clipitem, 'duration').text = str(telop['end_frame'] - telop['start_frame'])
                ET.SubElement(telop_clipitem, 'start').text = str(telop['start_frame'])
                ET.SubElement(telop_clipitem, 'end').text = str(telop['end_frame'])
                ET.SubElement(telop_clipitem, 'in').text = '0'
                ET.SubElement(telop_clipitem, 'out').text = str(telop['end_frame'] - telop['start_frame'])
                
                # グラフィックファイル
                telop_file = ET.SubElement(telop_clipitem, 'file', id=f"telop_file_{telop_idx}")
                ET.SubElement(telop_file, 'name').text = 'グラフィック'
                ET.SubElement(telop_file, 'mediaSource').text = 'GraphicAndType'
                
                # テロップエフェクト
                telop_filter = ET.SubElement(telop_clipitem, 'filter')
                telop_effect = ET.SubElement(telop_filter, 'effect')
                ET.SubElement(telop_effect, 'name').text = telop['text']
                ET.SubElement(telop_effect, 'effectid').text = 'GraphicAndType'
                ET.SubElement(telop_effect, 'effectcategory').text = 'graphic'
                ET.SubElement(telop_effect, 'effecttype').text = 'filter'
                ET.SubElement(telop_effect, 'mediatype').text = 'video'
        
        # XMLを整形して文字列化
        xml_str = minidom.parseString(ET.tostring(root, encoding='utf-8')).toprettyxml(indent="  ")
        return xml_str
    
    def _add_motion_effect(self, clipitem: ET.Element, scale: float, pos_x: float, pos_y: float):
        """モーションエフェクトを追加（csv2xml3.pyのパターン）"""
        filt = ET.SubElement(clipitem, 'filter')
        eff = ET.SubElement(filt, 'effect')
        ET.SubElement(eff, 'name').text = "Basic Motion"
        ET.SubElement(eff, 'effectid').text = "basic"
        
        # スケール
        ps = ET.SubElement(eff, 'parameter')
        ET.SubElement(ps, 'parameterid').text = "scale"
        ET.SubElement(ps, 'name').text = "Scale"
        ET.SubElement(ps, 'value').text = str(int(scale * 100))  # パーセンテージに変換
        
        # ポジション
        pc = ET.SubElement(eff, 'parameter')
        ET.SubElement(pc, 'parameterid').text = "center"
        ET.SubElement(pc, 'name').text = "Center"
        val = ET.SubElement(pc, 'value')
        ET.SubElement(val, 'horiz').text = str(pos_x)
        ET.SubElement(val, 'vert').text = str(pos_y)
    
    def _add_crop_effect(self, clipitem: ET.Element, left: float, right: float, top: float, bottom: float):
        """クロップエフェクトを追加（csv2xml3.pyのパターン）"""
        filt = ET.SubElement(clipitem, 'filter')
        eff = ET.SubElement(filt, 'effect')
        ET.SubElement(eff, 'name').text = "Crop"
        ET.SubElement(eff, 'effectid').text = "Crop"
        
        params = {'left': left, 'right': right, 'top': top, 'bottom': bottom}
        for key, val in params.items():
            p = ET.SubElement(eff, 'parameter')
            ET.SubElement(p, 'parameterid').text = key
            ET.SubElement(p, 'name').text = key.capitalize()
            ET.SubElement(p, 'valuemin').text = "0"
            ET.SubElement(p, 'valuemax').text = "100"
            ET.SubElement(p, 'value').text = str(int(val))


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="動画編集推論パイプライン")
    
    parser.add_argument('video_path', type=str,
                       help='入力動画ファイルのパス')
    parser.add_argument('--model', type=str, default='checkpoints_experiment/best_model.pth',
                       help='学習済みモデルのパス')
    parser.add_argument('--output', type=str, default=None,
                       help='出力XMLファイルのパス（デフォルト: {video_name}_edited.xml）')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='使用デバイス')
    parser.add_argument('--fps', type=float, default=10.0,
                       help='フレームレート')
    parser.add_argument('--num_tracks', type=int, default=20,
                       help='トラック数')
    parser.add_argument('--telop_config', type=str, default=None,
                       help='テロップ生成設定ファイルのパス（デフォルト: configs/config_telop_generation.yaml）')
    parser.add_argument('--no-speech', action='store_true',
                       help='音声認識を無効化')
    parser.add_argument('--no-emotion', action='store_true',
                       help='感情検出を無効化')
    
    args = parser.parse_args()
    
    # 出力パスを決定
    if args.output is None:
        video_name = Path(args.video_path).stem
        args.output = f"{video_name}_edited.xml"
    
    # パイプラインを実行
    pipeline = InferencePipeline(
        model_path=args.model,
        device=args.device,
        fps=args.fps,
        num_tracks=args.num_tracks,
        telop_config_path=args.telop_config
    )
    
    # コマンドライン引数で機能を無効化
    if args.no_speech:
        pipeline.telop_config.config['telop_generation']['speech']['enabled'] = False
        logger.info("Speech recognition disabled by command-line argument")
    if args.no_emotion:
        pipeline.telop_config.config['telop_generation']['emotion']['enabled'] = False
        logger.info("Emotion detection disabled by command-line argument")
    
    output_xml = pipeline.predict(
        video_path=args.video_path,
        output_xml_path=args.output
    )
    
    logger.info(f"\n✅ 完了！")
    logger.info(f"出力XML: {output_xml}")
    logger.info(f"\nPremiere Proで開いて編集を確認してください。")


if __name__ == "__main__":
    main()
