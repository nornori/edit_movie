# Design Document

## Overview

This design document describes the architecture and implementation approach for AI-powered telop generation. The system extends the existing inference pipeline to automatically generate subtitles from speech recognition and emotion-based text overlays from audio analysis. The generated telops are output as Premiere Pro-compatible graphics in the XML timeline.

The design follows these principles:
1. **Minimal changes to existing code** - Extend existing files rather than creating new modules
2. **Modular emotion detection** - Easy to add new emotions in the future
3. **Configurable behavior** - Enable/disable features via configuration
4. **Backward compatibility** - System works without ASR/emotion models installed

## Architecture

### High-Level Flow

```
Video Input
    ↓
[Feature Extraction] (existing)
    ↓
[Speech Recognition] (NEW) ──→ Speech Telops
    ↓
[Emotion Detection] (NEW) ──→ Emotion Telops
    ↓
[Model Prediction] (existing)
    ↓
[XML Generation] (modified) ──→ Video + Audio + OCR Telops + AI Telops
    ↓
[Telop Post-Processing] (modified)
    ↓
Premiere Pro XML Output
```

### Component Modifications

1. **inference_pipeline.py**
   - Add `_extract_speech_telops()` method
   - Add `_detect_emotion_telops()` method
   - Modify `_create_xml()` to include AI-generated telops
   - Add configuration parameters for ASR and emotion detection

2. **otio_xml_generator.py**
   - Modify `create_premiere_xml_with_otio()` to accept `ai_telops` parameter
   - Distinguish between OCR telops and AI telops using markers
   - Create separate tracks for different telop types

3. **fix_telop_simple.py**
   - Extend to handle both `[Telop]` (OCR) and `[AI-Telop]` markers
   - Apply same graphic conversion to AI-generated telops

## Components and Interfaces

### 1. Speech Recognition Module

**Location**: `inference_pipeline.py` - new method `_extract_speech_telops()`

**Interface**:
```python
def _extract_speech_telops(self, video_path: str, video_name: str) -> list:
    """
    Extract speech and generate telops using ASR
    
    Args:
        video_path: Path to video file
        video_name: Video name for caching
    
    Returns:
        List of dicts with keys: 'text', 'start_frame', 'end_frame', 'type'='speech'
    """
```

**Dependencies**:
- Whisper (openai-whisper) or Vosk for ASR
- Audio extraction using librosa or ffmpeg

**Implementation Strategy**:
- Use Whisper small model for Japanese/English
- Cache ASR results in temp_features directory
- Segment long transcriptions at sentence boundaries
- Merge short segments (< 0.5s) with adjacent segments

### 2. Emotion Detection Module

**Location**: `inference_pipeline.py` - new method `_detect_emotion_telops()`

**Interface**:
```python
def _detect_emotion_telops(self, video_path: str, video_name: str) -> list:
    """
    Detect emotions and generate telops
    
    Args:
        video_path: Path to video file
        video_name: Video name for caching
    
    Returns:
        List of dicts with keys: 'text', 'start_frame', 'end_frame', 'type'='emotion', 'emotion_type'
    """
```

**Emotion Types**:
- `laughter`: High pitch variation + energy bursts → "www" or "笑"
- `surprise`: Sudden pitch increase + short duration → "！" or "えっ"
- `sadness`: Low pitch + low energy + slow rate → "..." or "悲"

**Implementation Strategy**:
- Extract audio features: MFCCs, pitch (F0), energy (RMS), zero-crossing rate
- Use rule-based classifiers for each emotion:
  - Laughter: pitch_std > threshold AND energy > threshold
  - Surprise: pitch_delta > threshold AND duration < 1s
  - Sadness: pitch_mean < threshold AND energy < threshold
- Return segments with confidence scores
- Filter by confidence threshold (default 0.6)

### 3. XML Generation Extension

**Location**: `otio_xml_generator.py` - modify `create_premiere_xml_with_otio()`

**Changes**:
```python
def create_premiere_xml_with_otio(
    video_path: str,
    video_name: str,
    total_frames: int,
    fps: float,
    tracks_data: list,
    telops: list,  # OCR telops
    ai_telops: list,  # NEW: AI-generated telops
    output_path: str
) -> str:
```

**Telop Markers**:
- OCR telops: `[Telop] {text}`
- AI speech telops: `[AI-Speech] {text}`
- AI emotion telops: `[AI-Emotion] {text}`

**Track Organization**:
```
Video Track 1: Main video
Video Track 2: OCR Telop 1
Video Track 3: OCR Telop 2
...
Video Track N: AI Speech Telop 1
Video Track N+1: AI Speech Telop 2
...
Video Track M: AI Emotion Telop 1
Video Track M+1: AI Emotion Telop 2
...
Audio Track 1: Audio
```

### 4. Telop Post-Processing Extension

**Location**: `fix_telop_simple.py` - modify `fix_telops()`

**Changes**:
- Detect `[Telop]`, `[AI-Speech]`, and `[AI-Emotion]` markers
- Apply same graphic conversion to all telop types
- Use shared file element for all telops (first telop defines, rest reference)

## Data Models

### Telop Data Structure

```python
{
    'text': str,           # Display text
    'start_frame': int,    # Start frame (in inference FPS)
    'end_frame': int,      # End frame (in inference FPS)
    'type': str,           # 'ocr', 'speech', or 'emotion'
    'emotion_type': str,   # Optional: 'laughter', 'surprise', 'sadness'
    'confidence': float    # Optional: 0.0 to 1.0
}
```

### Configuration Structure

```yaml
# config_inference.yaml (NEW)
telop_generation:
  enabled: true
  
  speech:
    enabled: true
    model: "whisper"  # or "vosk"
    model_size: "small"  # tiny, small, medium, large
    language: "ja"  # ja, en
    min_segment_duration: 0.5  # seconds
    max_segment_duration: 5.0  # seconds
  
  emotion:
    enabled: true
    confidence_threshold: 0.6
    
    laughter:
      enabled: true
      text_short: "w"
      text_medium: "www"
      text_long: "wwww"
      pitch_std_threshold: 50.0
      energy_threshold: 0.3
    
    surprise:
      enabled: true
      text: "！"
      pitch_delta_threshold: 100.0
      max_duration: 1.0
    
    sadness:
      enabled: true
      text: "..."
      pitch_mean_threshold: 150.0
      energy_threshold: 0.1
```

## Error Handling

### ASR Failures
- **Model not installed**: Log warning, skip speech telops, continue
- **Audio extraction fails**: Log error, skip speech telops, continue
- **Transcription fails**: Log error with timestamp, skip that segment
- **Empty transcription**: No telops generated, log info

### Emotion Detection Failures
- **Feature extraction fails**: Log error, skip emotion telops, continue
- **No emotions detected**: No telops generated, log info
- **Low confidence**: Filter out, log debug message

### XML Generation Failures
- **Telop overlap**: Create separate tracks, log warning
- **Invalid timestamps**: Clip to video duration, log warning
- **Empty text**: Skip telop, log warning

## Testing Strategy

### Unit Tests

1. **test_speech_recognition.py**
   - Test ASR with sample audio files
   - Test segment merging logic
   - Test segment splitting logic
   - Test caching mechanism

2. **test_emotion_detection.py**
   - Test laughter detection with synthetic audio
   - Test surprise detection with pitch jumps
   - Test sadness detection with low-energy audio
   - Test confidence filtering

3. **test_xml_generation.py**
   - Test AI telop addition to XML
   - Test telop marker handling
   - Test track organization
   - Test backward compatibility (no AI telops)

4. **test_telop_postprocessing.py**
   - Test AI telop graphic conversion
   - Test marker detection
   - Test file element sharing

### Integration Tests

1. **test_end_to_end.py**
   - Process sample video with speech and emotions
   - Verify XML contains all telop types
   - Verify Premiere Pro can import XML
   - Verify telops are editable in Premiere Pro

### Manual Testing

1. Test with real YouTube video
2. Verify speech telops match spoken words
3. Verify emotion telops appear at correct times
4. Verify telops are editable in Premiere Pro
5. Test with different languages (Japanese, English)


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Speech segment to telop mapping
*For any* set of speech segments detected by ASR, the number of generated speech telops should equal the number of segments (after merging/splitting).
**Validates: Requirements 1.3**

### Property 2: Short segment merging
*For any* sequence of speech segments where adjacent segments are both shorter than 0.5 seconds, they should be merged into a single telop.
**Validates: Requirements 1.4**

### Property 3: Long segment splitting
*For any* speech segment longer than 5 seconds, it should be split into multiple telops at natural pauses (sentence boundaries).
**Validates: Requirements 1.5**

### Property 4: Emotion text mapping
*For any* detected emotion, the generated telop text should match the configured text pattern for that emotion type (laughter → "www", surprise → "！", sadness → "...").
**Validates: Requirements 2.2, 2.3, 2.4**

### Property 5: Emotion-speech track separation
*For any* pair of overlapping speech and emotion telops, they should be placed on separate video tracks in the XML.
**Validates: Requirements 2.5**

### Property 6: ASR output format
*For any* successful transcription, the output should contain text segments with both start_time and end_time timestamps.
**Validates: Requirements 3.3**

### Property 7: Audio sample rate consistency
*For any* video processed, the extracted audio sample rate should match the ASR model's required sample rate.
**Validates: Requirements 3.2**

### Property 8: Emotion detection output format
*For any* emotion detection result, it should include timestamps, emotion type, and confidence score.
**Validates: Requirements 4.5**

### Property 9: Confidence filtering
*For any* emotion detection with confidence below the threshold (default 0.6), no telop should be generated.
**Validates: Requirements 4.6**

### Property 10: Graphic layer format
*For any* generated telop in the XML, the file element should have mediaSource="GraphicAndType".
**Validates: Requirements 5.1**

### Property 11: File element sharing
*For any* XML with multiple telops, the first telop should have a complete file element, and subsequent telops should reference it by ID.
**Validates: Requirements 5.2**

### Property 12: Telop text storage
*For any* telop in the XML, the text content should be stored in the clipitem's name field.
**Validates: Requirements 5.3**

### Property 13: Overlapping telop track separation
*For any* set of telops with overlapping time ranges, each telop should be on a separate video track.
**Validates: Requirements 5.4**

### Property 14: Threshold configuration effect
*For any* emotion detection threshold value, increasing the threshold should decrease or maintain the number of detected emotions (never increase).
**Validates: Requirements 6.3**

### Property 15: Custom text application
*For any* custom emotion text pattern configured, all telops of that emotion type should use the custom text.
**Validates: Requirements 6.4**

### Property 16: Timestamp alignment
*For any* generated telop, the start and end timestamps should align with the corresponding audio event timestamps within tolerance (±0.1 seconds).
**Validates: Requirements 7.3**

### Property 17: XML telop format validation
*For any* generated XML with telops, all telop clipitems should have the required fields: name, start, end, in, out, and file reference.
**Validates: Requirements 7.4**

### Property 18: Telop generation independence
*For any* video, enabling or disabling telop generation should not affect the predicted video/audio track parameters.
**Validates: Requirements 8.5**

