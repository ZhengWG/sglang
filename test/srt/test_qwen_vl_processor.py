"""
Unit tests for QwenVLImageProcessor to ensure consistency with transformers implementations.
Tests the process_mm_data interface for image, video, and audio data processing.
"""

import math
import pytest
import torch
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image
import io

from sglang.srt.multimodal.processors.qwen_vl import (
    QwenVLImageProcessor,
    smart_nframes,
    smart_resize,
    detect_qwen_version,
    get_version_config,
)


class TestTransformersConsistency:
    """Test consistency with transformers implementations using exact same logic."""
    
    def test_qwen2_smart_resize_consistency(self):
        """Test that our smart_resize matches transformers Qwen2 implementation exactly."""
        # Import transformers implementation for comparison
        import sys
        sys.path.append('/Users/zhengwengang/Project/projects/work_projects/sglang/transformers/src')
        
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as transformers_smart_resize
        
        # Test cases with different aspect ratios and sizes
        test_cases = [
            (224, 224),   # Square image
            (480, 640),   # Portrait
            (640, 480),   # Landscape
            (100, 200),   # Very tall
            (200, 100),   # Very wide
        ]
        
        for height, width in test_cases:
            # Our implementation
            our_h, our_w = smart_resize(height, width, version="qwen2_vl")
            
            # Transformers implementation
            transformers_h, transformers_w = transformers_smart_resize(
                height, width, 
                factor=28, 
                min_pixels=56*56, 
                max_pixels=14*14*4*1280
            )
            
            assert our_h == transformers_h, f"Height mismatch for {height}x{width}: our={our_h}, transformers={transformers_h}"
            assert our_w == transformers_w, f"Width mismatch for {height}x{width}: our={our_w}, transformers={transformers_w}"
    
    def test_qwen3_smart_resize_consistency(self):
        """Test that our smart_resize matches transformers Qwen3 implementation exactly."""
        # Import transformers implementation for comparison
        import sys
        sys.path.append('/Users/zhengwengang/Project/projects/work_projects/sglang/transformers/src')
        
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize as transformers_smart_resize
        
        # Test cases with different aspect ratios and sizes
        test_cases = [
            (224, 224),   # Square image
            (480, 640),   # Portrait
            (640, 480),   # Landscape
            (100, 200),   # Very tall
            (200, 100),   # Very wide
        ]
        
        for height, width in test_cases:
            num_frames = 10  # Qwen3 smart_resize requires num_frames
            
            # Our implementation (we need to adapt it for Qwen3)
            our_h, our_w = smart_resize(height, width, version="qwen3_vl")
            
            # Transformers implementation
            transformers_t, transformers_h, transformers_w = transformers_smart_resize(
                num_frames, height, width,
                temporal_factor=2,
                factor=32,
                min_pixels=128*128,
                max_pixels=16*16*2*2*2*6144
            )
            
            assert our_h == transformers_h, f"Height mismatch for {height}x{width}: our={our_h}, transformers={transformers_h}"
            assert our_w == transformers_w, f"Width mismatch for {height}x{width}: our={our_w}, transformers={transformers_w}"
    
    def test_qwen2_sample_frames_consistency(self):
        """Test that our smart_nframes matches transformers Qwen2 sample_frames exactly."""
        # Import transformers implementation for comparison
        import sys
        sys.path.append('/Users/zhengwengang/Project/projects/work_projects/sglang/transformers/src')
        
        from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor
        from transformers.video_utils import VideoMetadata
        
        # Create transformers processor
        transformers_processor = Qwen2VLVideoProcessor()
        
        # Test cases
        test_cases = [
            # (total_frames, video_fps, target_fps, expected_frames)
            (60, 30.0, 2.0, 4),    # 60/30*2 = 4
            (30, 30.0, 1.0, 4),    # 30/30*1 = 1, but min is 4
            (120, 30.0, 4.0, 8),   # 120/30*4 = 16, but max is 8 (adjusted for temporal_patch_size)
            (100, 25.0, 2.0, 8),   # 100/25*2 = 8
        ]
        
        for total_frames, video_fps, target_fps, expected in test_cases:
            # Create video metadata
            metadata = VideoMetadata(
                total_num_frames=total_frames,
                fps=video_fps,
                duration=total_frames / video_fps
            )
            
            # Transformers implementation
            transformers_indices = transformers_processor.sample_frames(
                metadata=metadata,
                fps=target_fps
            )
            transformers_frames = len(transformers_indices)
            
            # Our implementation
            ele = {"fps": target_fps}
            our_frames = smart_nframes(ele, total_frames, video_fps, version="qwen2_vl")
            
            assert our_frames == transformers_frames, f"Frame count mismatch for {total_frames}frames@{video_fps}fps->{target_fps}fps: our={our_frames}, transformers={transformers_frames}"
    
    def test_qwen3_sample_frames_consistency(self):
        """Test that our smart_nframes matches transformers Qwen3 sample_frames exactly."""
        # Import transformers implementation for comparison
        import sys
        sys.path.append('/Users/zhengwengang/Project/projects/work_projects/sglang/transformers/src')
        
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
        from transformers.video_utils import VideoMetadata
        
        # Create transformers processor
        transformers_processor = Qwen3VLVideoProcessor()
        
        # Test cases
        test_cases = [
            # (total_frames, video_fps, target_fps, expected_frames)
            (60, 30.0, 2.0, 4),    # int(60/30*2) = 4
            (30, 30.0, 1.0, 4),    # int(30/30*1) = 1, but min is 4
            (120, 30.0, 4.0, 16),  # int(120/30*4) = 16
            (100, 25.0, 2.0, 8),   # int(100/25*2) = 8
        ]
        
        for total_frames, video_fps, target_fps, expected in test_cases:
            # Create video metadata
            metadata = VideoMetadata(
                total_num_frames=total_frames,
                fps=video_fps,
                duration=total_frames / video_fps
            )
            
            # Transformers implementation
            transformers_indices = transformers_processor.sample_frames(
                metadata=metadata,
                fps=target_fps
            )
            transformers_frames = len(transformers_indices)
            
            # Our implementation
            ele = {"fps": target_fps}
            our_frames = smart_nframes(ele, total_frames, video_fps, version="qwen3_vl")
            
            assert our_frames == transformers_frames, f"Frame count mismatch for {total_frames}frames@{video_fps}fps->{target_fps}fps: our={our_frames}, transformers={transformers_frames}"
    
    def test_explicit_nframes_consistency(self):
        """Test that explicit nframes handling matches transformers."""
        # Test cases with explicit nframes
        test_cases = [
            ({"nframes": 10}, 100, 30.0, 10),
            ({"nframes": 5}, 50, 25.0, 5),
            ({"nframes": 20}, 200, 30.0, 20),
        ]
        
        for ele, total_frames, video_fps, expected in test_cases:
            # Qwen2 implementation
            qwen2_frames = smart_nframes(ele, total_frames, video_fps, version="qwen2_vl")
            # Qwen2 should round to temporal_patch_size (2)
            expected_qwen2 = math.floor(expected / 2) * 2
            assert qwen2_frames == expected_qwen2, f"Qwen2 nframes mismatch: expected={expected_qwen2}, got={qwen2_frames}"
            
            # Qwen3 implementation
            qwen3_frames = smart_nframes(ele, total_frames, video_fps, version="qwen3_vl")
            # Qwen3 should use exact nframes
            assert qwen3_frames == expected, f"Qwen3 nframes mismatch: expected={expected}, got={qwen3_frames}"


class TestProcessMMDataInterface:
    """Test the main process_mm_data interface for different data types."""
    
    def create_mock_processor(self, version="qwen2_vl"):
        """Create a mock QwenVLImageProcessor for testing."""
        hf_config = Mock()
        hf_config.model_type = version
        hf_config.vision_start_token_id = 1
        hf_config.vision_end_token_id = 2
        hf_config.image_token_id = 3
        hf_config.video_token_id = 4
        
        server_args = Mock()
        _processor = Mock()
        
        with patch('sglang.srt.multimodal.processors.qwen_vl.SGLangBaseProcessor.__init__'):
            processor = QwenVLImageProcessor(hf_config, server_args, _processor)
        
        return processor
    
    def create_test_image(self, width=224, height=224):
        """Create a test PIL Image."""
        return Image.new('RGB', (width, height), color='red')
    
    def create_test_video_reader(self, total_frames=60, fps=30.0):
        """Create a mock video reader."""
        vr = Mock()
        vr.get_meta_data.return_value = {
            'fps': fps,
            'nframes': total_frames,
            'width': 640,
            'height': 480
        }
        vr.__len__.return_value = total_frames
        return vr
    
    @pytest.mark.asyncio
    async def test_process_image_data_qwen2(self):
        """Test image processing with Qwen2."""
        processor = self.create_mock_processor("qwen2_vl")
        test_image = self.create_test_image(224, 224)
        
        # Mock the resize_image_async method
        with patch.object(processor, 'resize_image_async', new_callable=AsyncMock) as mock_resize:
            mock_resize.return_value = test_image
            
            result = await processor.process_mm_data_async(test_image, "image")
            
            # Verify resize_image_async was called with correct parameters
            mock_resize.assert_called_once()
            call_args = mock_resize.call_args
            assert call_args[0][0] == test_image  # image
            assert call_args[1]['version'] == "qwen2_vl"
    
    @pytest.mark.asyncio
    async def test_process_image_data_qwen3(self):
        """Test image processing with Qwen3."""
        processor = self.create_mock_processor("qwen3_vl")
        test_image = self.create_test_image(224, 224)
        
        # Mock the resize_image_async method
        with patch.object(processor, 'resize_image_async', new_callable=AsyncMock) as mock_resize:
            mock_resize.return_value = test_image
            
            result = await processor.process_mm_data_async(test_image, "image")
            
            # Verify resize_image_async was called with correct parameters
            mock_resize.assert_called_once()
            call_args = mock_resize.call_args
            assert call_args[0][0] == test_image  # image
            assert call_args[1]['version'] == "qwen3_vl"
    
    @pytest.mark.asyncio
    async def test_process_video_data_qwen2(self):
        """Test video processing with Qwen2."""
        processor = self.create_mock_processor("qwen2_vl")
        test_vr = self.create_test_video_reader(total_frames=60, fps=30.0)
        
        # Mock the preprocess_video method
        with patch.object(processor, 'preprocess_video', new_callable=AsyncMock) as mock_preprocess:
            mock_preprocess.return_value = (torch.randn(4, 3, 224, 224), 4)
            
            result = await processor.process_mm_data_async(test_vr, "video")
            
            # Verify preprocess_video was called with correct parameters
            mock_preprocess.assert_called_once()
            call_args = mock_preprocess.call_args
            assert call_args[0][0] == test_vr  # video reader
            assert call_args[1]['version'] == "qwen2_vl"
    
    @pytest.mark.asyncio
    async def test_process_video_data_qwen3(self):
        """Test video processing with Qwen3."""
        processor = self.create_mock_processor("qwen3_vl")
        test_vr = self.create_test_video_reader(total_frames=60, fps=30.0)
        
        # Mock the preprocess_video method
        with patch.object(processor, 'preprocess_video', new_callable=AsyncMock) as mock_preprocess:
            mock_preprocess.return_value = (torch.randn(4, 3, 224, 224), 4)
            
            result = await processor.process_mm_data_async(test_vr, "video")
            
            # Verify preprocess_video was called with correct parameters
            mock_preprocess.assert_called_once()
            call_args = mock_preprocess.call_args
            assert call_args[0][0] == test_vr  # video reader
            assert call_args[1]['version'] == "qwen3_vl"
    
    @pytest.mark.asyncio
    async def test_process_audio_data(self):
        """Test audio processing (should return None)."""
        processor = self.create_mock_processor("qwen2_vl")
        test_audio = np.random.randn(16000)  # 1 second of audio at 16kHz
        
        result = await processor.process_mm_data_async(test_audio, "audio")
        
        # Audio should return None (not supported)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_process_unsupported_data_type(self):
        """Test processing of unsupported data type."""
        processor = self.create_mock_processor("qwen2_vl")
        test_data = "some text data"
        
        result = await processor.process_mm_data_async(test_data, "text")
        
        # Unsupported data type should return None
        assert result is None


class TestVersionDetection:
    """Test Qwen version detection functionality."""
    
    def test_detect_qwen2_vl(self):
        """Test detection of qwen2_vl model."""
        config = Mock()
        config.model_type = "qwen2_vl"
        assert detect_qwen_version(config) == "qwen2_vl"
    
    def test_detect_qwen2_5_vl(self):
        """Test detection of qwen2_5_vl model."""
        config = Mock()
        config.model_type = "qwen2_5_vl"
        assert detect_qwen_version(config) == "qwen2_5_vl"
    
    def test_detect_qwen3_vl(self):
        """Test detection of qwen3_vl model."""
        config = Mock()
        config.model_type = "qwen3_vl"
        assert detect_qwen_version(config) == "qwen3_vl"
    
    def test_detect_qwen3_vl_moe(self):
        """Test detection of qwen3_vl_moe model."""
        config = Mock()
        config.model_type = "qwen3_vl_moe"
        assert detect_qwen_version(config) == "qwen3_vl_moe"
    
    def test_detect_from_architecture(self):
        """Test detection from architecture name when model_type is not available."""
        config = Mock()
        config.model_type = "unknown"
        config.architectures = ["Qwen2VLForConditionalGeneration"]
        assert detect_qwen_version(config) == "qwen2_vl"
        
        config.architectures = ["Qwen2_5_VLForConditionalGeneration"]
        assert detect_qwen_version(config) == "qwen2_5_vl"
        
        config.architectures = ["Qwen3VLForConditionalGeneration"]
        assert detect_qwen_version(config) == "qwen3_vl"
        
        config.architectures = ["Qwen3VLMoeForConditionalGeneration"]
        assert detect_qwen_version(config) == "qwen3_vl_moe"


class TestErrorHandling:
    """Test error handling in the processor."""
    
    def test_invalid_version_handling(self):
        """Test handling of invalid Qwen versions."""
        with pytest.raises(ValueError, match="Unsupported Qwen version"):
            smart_nframes({"fps": 2.0}, 60, 30.0, version="invalid_version")
    
    def test_mutually_exclusive_fps_nframes(self):
        """Test that fps and nframes cannot be specified together."""
        ele = {"fps": 2.0, "nframes": 10}
        
        with pytest.raises(AssertionError, match="Only accept either"):
            smart_nframes(ele, 60, 30.0, version="qwen2_vl")
    
    def test_aspect_ratio_validation(self):
        """Test aspect ratio validation in smart_resize."""
        with pytest.raises(ValueError, match="absolute aspect ratio must be smaller than"):
            smart_resize(1000, 1, version="qwen2_vl")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])