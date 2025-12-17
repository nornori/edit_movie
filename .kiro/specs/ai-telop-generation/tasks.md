# Implementation Plan

- [x] 1. Set up ASR and audio processing dependencies


  - Install Whisper and audio processing libraries (openai-whisper, librosa)
  - Create configuration file structure for telop generation settings
  - Add configuration loading to inference pipeline
  - _Requirements: 3.1, 6.1, 6.5_

- [ ]* 1.1 Write property test for configuration loading
  - **Property 14: Threshold configuration effect**
  - **Validates: Requirements 6.3**



- [ ] 2. Implement speech recognition module in inference_pipeline.py
  - Add `_extract_speech_telops()` method to InferencePipeline class
  - Implement audio extraction from video using librosa
  - Integrate Whisper ASR model for transcription
  - Add caching mechanism for ASR results in temp_features directory
  - _Requirements: 1.1, 1.2, 3.2, 3.3_

- [ ]* 2.1 Write property test for ASR output format
  - **Property 6: ASR output format**
  - **Validates: Requirements 3.3**

- [x]* 2.2 Write property test for audio sample rate


  - **Property 7: Audio sample rate consistency**
  - **Validates: Requirements 3.2**

- [ ] 3. Implement speech segment processing logic
  - Add segment merging for short segments (< 0.5s)
  - Add segment splitting for long segments (> 5s) at sentence boundaries
  - Implement telop generation from speech segments
  - _Requirements: 1.3, 1.4, 1.5_

- [ ]* 3.1 Write property test for speech-to-telop mapping
  - **Property 1: Speech segment to telop mapping**
  - **Validates: Requirements 1.3**

- [ ]* 3.2 Write property test for short segment merging
  - **Property 2: Short segment merging**
  - **Validates: Requirements 1.4**



- [ ]* 3.3 Write property test for long segment splitting
  - **Property 3: Long segment splitting**
  - **Validates: Requirements 1.5**

- [ ] 4. Implement emotion detection module in inference_pipeline.py
  - Add `_detect_emotion_telops()` method to InferencePipeline class
  - Implement audio feature extraction (MFCCs, pitch, energy, zero-crossing rate)
  - Implement laughter detection using pitch variation and energy bursts
  - Implement surprise detection using sudden pitch increases
  - Implement sadness detection using low pitch and energy
  - Add confidence scoring and filtering
  - _Requirements: 2.1, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [ ]* 4.1 Write property test for emotion detection output format
  - **Property 8: Emotion detection output format**
  - **Validates: Requirements 4.5**

- [x]* 4.2 Write property test for confidence filtering


  - **Property 9: Confidence filtering**
  - **Validates: Requirements 4.6**

- [ ]* 4.3 Write property test for emotion text mapping
  - **Property 4: Emotion text mapping**
  - **Validates: Requirements 2.2, 2.3, 2.4**

- [ ] 5. Integrate AI telops into XML generation
  - Modify `_create_xml()` in inference_pipeline.py to call speech and emotion detection
  - Pass AI telops to `create_premiere_xml_with_otio()` in otio_xml_generator.py
  - Modify `create_premiere_xml_with_otio()` to accept `ai_telops` parameter
  - Add telop markers: `[AI-Speech]` and `[AI-Emotion]`
  - Implement track organization for OCR, speech, and emotion telops
  - _Requirements: 2.5, 5.1, 5.2, 5.3, 5.4_

- [ ]* 5.1 Write property test for emotion-speech track separation
  - **Property 5: Emotion-speech track separation**
  - **Validates: Requirements 2.5**

- [ ]* 5.2 Write property test for graphic layer format
  - **Property 10: Graphic layer format**
  - **Validates: Requirements 5.1**

- [ ]* 5.3 Write property test for file element sharing
  - **Property 11: File element sharing**
  - **Validates: Requirements 5.2**



- [ ]* 5.4 Write property test for telop text storage
  - **Property 12: Telop text storage**
  - **Validates: Requirements 5.3**

- [ ]* 5.5 Write property test for overlapping telop tracks
  - **Property 13: Overlapping telop track separation**
  - **Validates: Requirements 5.4**



- [ ] 6. Extend telop post-processing in fix_telop_simple.py
  - Modify `fix_telops()` to detect `[AI-Speech]` and `[AI-Emotion]` markers
  - Apply graphic conversion to AI telops (same as OCR telops)
  - Ensure all telop types share the same file element structure
  - _Requirements: 5.1, 5.2_

- [ ]* 6.1 Write property test for custom text application
  - **Property 15: Custom text application**
  - **Validates: Requirements 6.4**



- [ ] 7. Implement error handling and backward compatibility
  - Add try-except blocks for ASR failures
  - Add fallback behavior when ASR model is unavailable
  - Add handling for videos with no audio
  - Add configuration flags to enable/disable features
  - Ensure telop generation doesn't affect video/audio predictions
  - _Requirements: 3.4, 3.5, 8.1, 8.2, 8.3, 8.5_

- [ ]* 7.1 Write property test for telop generation independence
  - **Property 18: Telop generation independence**



  - **Validates: Requirements 8.5**

- [ ] 8. Add configuration and customization features
  - Create default configuration YAML file
  - Add command-line arguments for telop generation settings
  - Implement custom emotion text patterns from configuration
  - Add language selection for ASR (Japanese/English)
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ]* 8.1 Write unit tests for configuration loading
  - Test YAML parsing
  - Test default values
  - Test configuration validation
  - _Requirements: 6.5_

- [ ] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ]* 10. Create integration tests
  - Test end-to-end pipeline with sample video
  - Test speech telop generation
  - Test emotion telop generation
  - Test XML output format
  - Test Premiere Pro compatibility
  - _Requirements: 7.3, 7.4_

- [ ]* 10.1 Write property test for timestamp alignment
  - **Property 16: Timestamp alignment**
  - **Validates: Requirements 7.3**

- [ ]* 10.2 Write property test for XML format validation
  - **Property 17: XML telop format validation**
  - **Validates: Requirements 7.4**

- [ ]* 11. Create documentation and examples
  - Write usage guide for AI telop generation
  - Create example configuration files
  - Document emotion detection parameters
  - Add troubleshooting section
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 12. Final Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
