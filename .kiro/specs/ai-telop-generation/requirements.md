# Requirements Document

## Introduction

This document specifies the requirements for AI-powered telop (subtitle) generation in the video editing inference pipeline. The system will automatically generate text overlays based on speech recognition and emotion detection, outputting them as Premiere Pro-compatible graphics in the XML timeline. This enables automatic subtitle creation without manual transcription.

**Note on OCR vs AI Telops**: The existing system uses OCR to extract telops that are already burned into the video (visible text on screen). This new feature generates NEW telops based on audio analysis (speech + emotions), which are separate from OCR-extracted telops. Both can coexist in the output XML.

## Glossary

- **Telop**: Text overlay displayed on video, commonly used for subtitles or captions in Japanese video editing
- **Speech Recognition**: Automatic transcription of spoken words from audio using ASR (Automatic Speech Recognition)
- **Emotion Detection**: Analysis of audio/visual features to detect emotional states such as laughter
- **Graphic Layer**: Premiere Pro's text/graphic track type that allows editable text overlays
- **Inference Pipeline**: The system that processes new videos and generates editing XMLs
- **ASR Model**: Automatic Speech Recognition model (e.g., Whisper, Vosk) that converts speech to text
- **Emotion Detection**: Audio analysis to identify emotional states (laughter, surprise, sadness) based on pitch, energy, and spectral features

## Requirements

### Requirement 1

**User Story:** As a video editor, I want the system to automatically transcribe speech and create subtitle telops, so that I don't have to manually type subtitles.

#### Acceptance Criteria

1. WHEN the system processes a video THEN the system SHALL extract audio and perform speech recognition
2. WHEN speech is detected THEN the system SHALL transcribe the spoken words with timestamps
3. WHEN creating telops THEN the system SHALL generate one telop per speech segment with the transcribed text
4. WHEN speech segments are short (less than 0.5 seconds) THEN the system SHALL merge them with adjacent segments
5. WHEN speech segments are long (more than 5 seconds) THEN the system SHALL split them at natural pauses

### Requirement 2

**User Story:** As a content creator, I want the system to detect emotions (laughter, surprise, sadness) and add appropriate text, so that the emotional tone is visually represented.

#### Acceptance Criteria

1. WHEN analyzing audio THEN the system SHALL detect emotion segments using audio features (pitch variation, energy, spectral features)
2. WHEN laughter is detected THEN the system SHALL create a telop with text "www" or "笑" at that timestamp
3. WHEN surprise is detected THEN the system SHALL create a telop with text "！" or "えっ" at that timestamp
4. WHEN sadness is detected THEN the system SHALL create a telop with text "..." or "悲" at that timestamp
5. WHEN emotions overlap with speech THEN the system SHALL add emotion indicators as separate telops on different tracks

### Requirement 3

**User Story:** As a machine learning engineer, I want to integrate ASR models into the inference pipeline, so that speech recognition runs automatically during video processing.

#### Acceptance Criteria

1. WHEN initializing the pipeline THEN the system SHALL load a pre-trained ASR model (Whisper or Vosk)
2. WHEN processing video THEN the system SHALL extract audio at the appropriate sample rate for the ASR model
3. WHEN transcribing THEN the system SHALL return text segments with start_time and end_time timestamps
4. WHEN ASR fails THEN the system SHALL log the error and continue without speech telops
5. WHEN ASR model is unavailable THEN the system SHALL skip speech recognition and log a warning

### Requirement 4

**User Story:** As a developer, I want to detect multiple emotions using audio features, so that I can automatically generate emotion-based telops.

#### Acceptance Criteria

1. WHEN analyzing audio THEN the system SHALL compute spectral features (MFCCs, spectral centroid, zero-crossing rate, pitch)
2. WHEN detecting laughter THEN the system SHALL identify segments with high pitch variation and energy bursts
3. WHEN detecting surprise THEN the system SHALL identify segments with sudden pitch increases and short duration
4. WHEN detecting sadness THEN the system SHALL identify segments with low pitch, low energy, and slow speech rate
5. WHEN classifying emotions THEN the system SHALL use a threshold-based or ML-based classifier and return timestamps with confidence scores
6. WHEN confidence is low (below 0.6) THEN the system SHALL not generate an emotion telop

### Requirement 5

**User Story:** As a video editor, I want telops to be output as Premiere Pro graphics, so that I can edit the text directly in Premiere Pro.

#### Acceptance Criteria

1. WHEN generating XML THEN the system SHALL create telop tracks as graphic layers with mediaSource "GraphicAndType"
2. WHEN adding telops THEN the system SHALL use the first telop's complete file element and reference it for subsequent telops
3. WHEN setting telop text THEN the system SHALL use the clipitem name field to store the text content
4. WHEN telops overlap THEN the system SHALL create separate tracks for each telop to avoid conflicts
5. WHEN exporting XML THEN the system SHALL ensure telops are compatible with Premiere Pro 2023 and later

### Requirement 6

**User Story:** As a system administrator, I want configurable telop generation settings, so that I can customize the behavior for different use cases.

#### Acceptance Criteria

1. WHEN configuring the system THEN the system SHALL accept flags to enable/disable speech telops and laughter telops independently
2. WHEN setting ASR language THEN the system SHALL support Japanese and English language models
3. WHEN adjusting sensitivity THEN the system SHALL accept emotion detection threshold parameters (0.0 to 1.0) for each emotion type
4. WHEN customizing text THEN the system SHALL accept custom emotion text patterns (e.g., laughter: "www"/"笑", surprise: "！"/"えっ", sadness: "..."/"悲")
5. WHEN saving settings THEN the system SHALL store configuration in a YAML file for reuse

### Requirement 7

**User Story:** As a quality assurance engineer, I want to validate telop generation accuracy, so that I can ensure the system produces correct subtitles.

#### Acceptance Criteria

1. WHEN testing speech recognition THEN the system SHALL report Word Error Rate (WER) on a validation set
2. WHEN testing emotion detection THEN the system SHALL report precision and recall on labeled emotion segments for each emotion type
3. WHEN validating timestamps THEN the system SHALL verify that telop start/end times align with audio events
4. WHEN checking XML output THEN the system SHALL validate that all telops are properly formatted as graphics
5. WHEN comparing with manual telops THEN the system SHALL compute overlap percentage and text similarity scores

### Requirement 8

**User Story:** As a developer, I want to maintain backward compatibility, so that existing videos without telops continue to work.

#### Acceptance Criteria

1. WHEN telop generation is disabled THEN the system SHALL process videos without creating telop tracks
2. WHEN ASR model is not installed THEN the system SHALL skip speech telops and continue with other processing
3. WHEN video has no audio THEN the system SHALL skip telop generation and log a message
4. WHEN loading old XMLs THEN the system SHALL support XMLs with and without telop tracks
5. WHEN comparing models THEN the system SHALL ensure telop generation does not affect video/audio track predictions
