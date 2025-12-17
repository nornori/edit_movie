"""
Full Video XML Parser - Extract track information for entire source video

This parser extracts editing information from Premiere Pro XML and creates labels
for the ENTIRE source video, marking which parts are used and which are cut.

Key difference from premiere_xml_parser.py:
- Old: Only extracts edited timeline (e.g., 73 seconds)
- New: Extracts full source video (e.g., 235 seconds) with cut information
"""
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FullVideoXMLParser:
    """Parser for Premiere Pro XML that generates labels for entire source video"""
    
    def __init__(self, max_tracks: int = 20, fps: float = 10.0):
        """
        Initialize Full Video XML Parser
        
        Args:
            max_tracks: Maximum number of tracks (default: 20)
            fps: Sampling rate for output (default: 10.0)
        """
        self.max_tracks = max_tracks
        self.fps = fps
        self.frame_duration = 1.0 / fps
        
        # Asset ID mapping
        self.asset_mapping = {}
        self.next_asset_id = 0
        
        # Sequence timebase
        self.timebase = 30.0
        
        logger.info(f"FullVideoXMLParser initialized: max_tracks={max_tracks}, fps={fps}")
    
    def parse_time(self, frames: str, timebase: float = None) -> float:
        """Convert frame count to seconds"""
        if not frames:
            return 0.0
        try:
            frame_count = int(frames)
            tb = timebase if timebase else self.timebase
            return frame_count / tb
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def get_asset_id(self, clip_name: str) -> int:
        """Get or create asset ID for a clip name"""
        if clip_name not in self.asset_mapping:
            self.asset_mapping[clip_name] = self.next_asset_id % 10
            self.next_asset_id += 1
        return self.asset_mapping[clip_name]
    
    def get_source_video_duration(self, video_path: str) -> float:
        """
        Get duration of source video file
        
        Args:
            video_path: Path to video file
        
        Returns:
            Duration in seconds
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                return duration
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get video duration: {e}")
            return 0.0
    
    def parse_premiere_xml_full(self, xml_path: str, source_video_path: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Parse Premiere Pro XML and generate labels for ENTIRE source video
        
        Args:
            xml_path: Path to Premiere Pro XML file
            source_video_path: Optional path to source video (if not in XML)
        
        Returns:
            Tuple of (DataFrame with full video labels, metadata dict)
        """
        logger.info(f"Parsing XML: {xml_path}")
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Find sequence
        sequence = root.find('.//sequence')
        if sequence is None:
            raise ValueError("No sequence found in XML")
        
        # Get timebase
        rate_elem = sequence.find('.//rate/timebase')
        if rate_elem is not None and rate_elem.text:
            self.timebase = float(rate_elem.text)
        
        # Find all video tracks
        video_tracks = sequence.findall('.//video/track')
        logger.info(f"Found {len(video_tracks)} video tracks")
        
        # Extract source video path from first clipitem
        source_path = source_video_path
        if not source_path:
            first_clip = sequence.find('.//video/track/clipitem')
            if first_clip is not None:
                file_elem = first_clip.find('.//file/pathurl')
                if file_elem is not None and file_elem.text:
                    from urllib.parse import unquote
                    url = file_elem.text
                    # Remove file://localhost/ prefix
                    if url.startswith('file://localhost/'):
                        url = url[17:]
                    source_path = unquote(url).replace('/', '\\')
        
        if not source_path:
            raise ValueError("Could not determine source video path")
        
        logger.info(f"Source video: {source_path}")
        
        # Get source video duration
        source_duration = self.get_source_video_duration(source_path)
        if source_duration == 0:
            raise ValueError(f"Could not get duration for source video: {source_path}")
        
        logger.info(f"Source video duration: {source_duration:.1f} seconds")
        
        # Calculate number of timesteps for full video
        num_timesteps = int(np.ceil(source_duration * self.fps))
        logger.info(f"Generating {num_timesteps} timesteps at {self.fps} FPS")
        
        # Initialize track data for ENTIRE video (all inactive by default)
        # Shape: (num_timesteps, max_tracks, 12 parameters)
        track_data = np.zeros((num_timesteps, self.max_tracks, 12), dtype=np.float32)
        
        # Process each track
        for track_idx, track in enumerate(video_tracks[:self.max_tracks]):
            logger.info(f"Processing track {track_idx + 1}/{min(len(video_tracks), self.max_tracks)}")
            
            # Find all clipitems in this track
            clipitems = track.findall('clipitem')
            
            for clip in clipitems:
                # Get clip timing in TIMELINE (edited sequence)
                start_elem = clip.find('start')
                end_elem = clip.find('end')
                
                # Get source timing (in original video)
                in_elem = clip.find('in')
                out_elem = clip.find('out')
                
                if None in [start_elem, end_elem, in_elem, out_elem]:
                    continue
                
                # Timeline position (where it appears in edited sequence)
                timeline_start = self.parse_time(start_elem.text)
                timeline_end = self.parse_time(end_elem.text)
                
                # Source position (which part of original video)
                source_start = self.parse_time(in_elem.text)
                source_end = self.parse_time(out_elem.text)
                
                # Convert source times to timestep indices
                source_start_idx = int(source_start * self.fps)
                source_end_idx = int(source_end * self.fps)
                
                # Clip bounds check
                source_start_idx = max(0, min(source_start_idx, num_timesteps - 1))
                source_end_idx = max(0, min(source_end_idx, num_timesteps))
                
                if source_start_idx >= source_end_idx:
                    continue
                
                # Get clip name and asset ID
                name_elem = clip.find('name')
                clip_name = name_elem.text if name_elem is not None and name_elem.text else f'clip_{track_idx}'
                asset_id = self.get_asset_id(clip_name)
                
                # Check if enabled
                enabled_elem = clip.find('enabled')
                enabled = enabled_elem is None or enabled_elem.text != 'FALSE'
                
                if not enabled:
                    continue
                
                # Extract transform parameters (simplified - no keyframes for now)
                scale = 1.0
                pos_x = 0.0
                pos_y = 0.0
                anchor_x = 0.0
                anchor_y = 0.0
                rotation = 0.0
                crop_l = 0.0
                crop_r = 0.0
                crop_t = 0.0
                crop_b = 0.0
                
                # Fill in track data for this clip's source range
                for t_idx in range(source_start_idx, source_end_idx):
                    track_data[t_idx, track_idx, 0] = 1.0  # active
                    track_data[t_idx, track_idx, 1] = float(asset_id)
                    track_data[t_idx, track_idx, 2] = scale
                    track_data[t_idx, track_idx, 3] = pos_x
                    track_data[t_idx, track_idx, 4] = pos_y
                    track_data[t_idx, track_idx, 5] = anchor_x
                    track_data[t_idx, track_idx, 6] = anchor_y
                    track_data[t_idx, track_idx, 7] = rotation
                    track_data[t_idx, track_idx, 8] = crop_l
                    track_data[t_idx, track_idx, 9] = crop_r
                    track_data[t_idx, track_idx, 10] = crop_t
                    track_data[t_idx, track_idx, 11] = crop_b
                
                logger.debug(f"  Clip: {clip_name}, source: {source_start:.1f}-{source_end:.1f}s, "
                           f"timeline: {timeline_start:.1f}-{timeline_end:.1f}s")
        
        # Convert to DataFrame
        rows = []
        param_names = ['active', 'asset_id', 'scale', 'pos_x', 'pos_y', 'anchor_x', 'anchor_y',
                      'rotation', 'crop_l', 'crop_r', 'crop_t', 'crop_b']
        
        for t_idx in range(num_timesteps):
            time = t_idx * self.frame_duration
            for track_idx in range(self.max_tracks):
                row = {'time': time, 'track': track_idx}
                for param_idx, param_name in enumerate(param_names):
                    row[param_name] = track_data[t_idx, track_idx, param_idx]
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Calculate statistics
        total_frames = num_timesteps
        used_frames = np.sum(track_data[:, :, 0] > 0.5)  # Count active frames
        cut_frames = total_frames - used_frames
        
        metadata = {
            'source_video_path': source_path,
            'source_duration': source_duration,
            'num_timesteps': num_timesteps,
            'fps': self.fps,
            'total_frames': total_frames,
            'used_frames': int(used_frames),
            'cut_frames': int(cut_frames),
            'cut_ratio': cut_frames / total_frames if total_frames > 0 else 0,
            'num_tracks': self.max_tracks,
            'asset_mapping': self.asset_mapping
        }
        
        logger.info(f"✅ Parsed full video:")
        logger.info(f"   Total frames: {total_frames}")
        logger.info(f"   Used frames: {used_frames} ({100 * (1 - metadata['cut_ratio']):.1f}%)")
        logger.info(f"   Cut frames: {cut_frames} ({100 * metadata['cut_ratio']:.1f}%)")
        
        return df, metadata


def main():
    """Test the parser"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse Premiere Pro XML for full source video")
    parser.add_argument('xml_path', help='Path to Premiere Pro XML file')
    parser.add_argument('--output', '-o', help='Output CSV path')
    parser.add_argument('--fps', type=float, default=10.0, help='Sampling FPS (default: 10.0)')
    parser.add_argument('--max_tracks', type=int, default=20, help='Max tracks (default: 20)')
    
    args = parser.parse_args()
    
    # Parse XML
    xml_parser = FullVideoXMLParser(max_tracks=args.max_tracks, fps=args.fps)
    df, metadata = xml_parser.parse_premiere_xml_full(args.xml_path)
    
    # Save to CSV
    if args.output:
        output_path = Path(args.output)
    else:
        xml_name = Path(args.xml_path).stem
        output_path = Path(f'{xml_name}_full_tracks.csv')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"💾 Saved to: {output_path}")
    logger.info(f"   Shape: {df.shape}")
    logger.info(f"   Columns: {list(df.columns)}")


if __name__ == '__main__':
    main()
