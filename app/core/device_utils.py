"""Device detection and management utilities."""
from __future__ import annotations
import os
import sys
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    """Information about the detected compute device."""
    device: str
    is_cuda_available: bool
    cuda_device_count: int
    cuda_visible_devices: str
    device_name: Optional[str] = None
    memory_gb: Optional[float] = None


class DeviceDetector:
    """Utility class for detecting and managing compute devices."""
    
    @staticmethod
    def detect_device(prefer_gpu: bool = True) -> DeviceInfo:
        """
        Detect the best available compute device.
        
        Args:
            prefer_gpu: Whether to prefer GPU over CPU if available
            
        Returns:
            DeviceInfo containing device information and selection
        """
        # Check if torch is available
        try:
            import torch
        except ImportError:
            # No torch available, fall back to CPU
            return DeviceInfo(
                device='cpu',
                is_cuda_available=False,
                cuda_device_count=0,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                device_name='CPU (PyTorch not available)'
            )
        
        # Get CUDA availability information
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count()
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        
        device_name = None
        memory_gb = None
        
        # Determine the best device to use
        if cuda_available and prefer_gpu and cuda_device_count > 0:
            device = 'cuda'
            try:
                device_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except Exception:
                device_name = 'CUDA Device'
        else:
            device = 'cpu'
            device_name = 'CPU'
            
        return DeviceInfo(
            device=device,
            is_cuda_available=cuda_available,
            cuda_device_count=cuda_device_count,
            cuda_visible_devices=cuda_visible_devices,
            device_name=device_name,
            memory_gb=memory_gb
        )
    
    @staticmethod
    def validate_device_string(device_str: str) -> Tuple[bool, str]:
        """
        Validate a device string and return validation result.
        
        Args:
            device_str: Device string to validate (e.g., 'cuda', 'cpu', 'cuda:0')
            
        Returns:
            Tuple of (is_valid, corrected_device_string)
        """
        try:
            import torch
        except ImportError:
            # Without torch, only CPU is valid
            return device_str.lower() == 'cpu', 'cpu'
        
        device_str = device_str.lower().strip()
        
        if device_str == 'cpu':
            return True, 'cpu'
        
        if device_str.startswith('cuda'):
            if not torch.cuda.is_available():
                return False, 'cpu'
            
            # Handle 'cuda' (default) or 'cuda:N' format
            if device_str == 'cuda':
                if torch.cuda.device_count() > 0:
                    return True, 'cuda'
                else:
                    return False, 'cpu'
            
            # Handle 'cuda:N' format
            if ':' in device_str:
                try:
                    device_num = int(device_str.split(':')[1])
                    if 0 <= device_num < torch.cuda.device_count():
                        return True, device_str
                    else:
                        return False, 'cpu'
                except (ValueError, IndexError):
                    return False, 'cpu'
        
        # Unknown device format, default to CPU
        return False, 'cpu'
    
    @staticmethod
    def get_device_info_string(device_info: DeviceInfo) -> str:
        """
        Get a human-readable string describing the device information.
        
        Args:
            device_info: DeviceInfo object
            
        Returns:
            Formatted string with device information
        """
        lines = []
        lines.append(f"Selected device: {device_info.device}")
        
        if device_info.device_name:
            lines.append(f"Device name: {device_info.device_name}")
            
        if device_info.memory_gb:
            lines.append(f"Device memory: {device_info.memory_gb:.1f} GB")
            
        lines.append(f"CUDA available: {device_info.is_cuda_available}")
        lines.append(f"CUDA device count: {device_info.cuda_device_count}")
        
        if device_info.cuda_visible_devices:
            lines.append(f"CUDA_VISIBLE_DEVICES: {device_info.cuda_visible_devices}")
        
        return " | ".join(lines)
    
    @staticmethod
    def log_device_selection(device_info: DeviceInfo, print_fn=print) -> None:
        """
        Log device selection information.
        
        Args:
            device_info: DeviceInfo object
            print_fn: Function to use for logging (default: print)
        """
        info_str = DeviceDetector.get_device_info_string(device_info)
        print_fn(f"Device Detection: {info_str}")
        
        # Add warning if CUDA was requested but not available
        if not device_info.is_cuda_available and device_info.cuda_visible_devices:
            print_fn("Warning: CUDA_VISIBLE_DEVICES is set but CUDA is not available")
        elif device_info.is_cuda_available and device_info.device == 'cpu':
            print_fn("Info: CUDA is available but CPU was selected/forced")


def detect_and_validate_device(prefer_gpu: bool = True, 
                              requested_device: Optional[str] = None) -> str:
    """
    Convenience function to detect and validate device selection.
    
    Args:
        prefer_gpu: Whether to prefer GPU over CPU if available
        requested_device: Specific device requested (overrides prefer_gpu)
        
    Returns:
        Valid device string that can be used safely
    """
    device_info = DeviceDetector.detect_device(prefer_gpu)
    
    if requested_device:
        is_valid, validated_device = DeviceDetector.validate_device_string(requested_device)
        if is_valid:
            return validated_device
        else:
            # Fall back to detected device if requested device is invalid
            return device_info.device
    
    return device_info.device