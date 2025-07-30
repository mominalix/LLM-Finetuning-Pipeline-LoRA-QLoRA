"""GPU utilities for optimization and monitoring.

This module provides functions for GPU detection, memory optimization,
and performance monitoring during training.

Example:
    >>> from src.utils.gpu import get_gpu_info, optimize_gpu_memory
    >>> gpu_info = get_gpu_info()
    >>> optimize_gpu_memory()
"""

import logging
import os
import gc
from typing import Dict, List, Optional, Any
import torch


logger = logging.getLogger(__name__)


def get_gpu_info() -> Dict[str, Any]:
    """Get comprehensive GPU information.
    
    Returns:
        Dictionary containing GPU information
        
    Example:
        >>> info = get_gpu_info()
        >>> print(f"Using {info['device_count']} GPUs")
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": [],
        "current_device": None,
        "cuda_version": None,
    }
    
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["current_device"] = torch.cuda.current_device()
        info["cuda_version"] = torch.version.cuda
        
        # Get information for each GPU
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            device_info = {
                "id": i,
                "name": device_props.name,
                "total_memory": device_props.total_memory,
                "total_memory_gb": device_props.total_memory / (1024**3),
                "major": device_props.major,
                "minor": device_props.minor,
                "multi_processor_count": device_props.multi_processor_count,
            }
            
            # Get current memory usage
            if i == torch.cuda.current_device() or torch.cuda.device_count() == 1:
                torch.cuda.empty_cache()  # Clear cache for accurate reading
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                
                device_info.update({
                    "memory_allocated": memory_allocated,
                    "memory_allocated_gb": memory_allocated / (1024**3),
                    "memory_reserved": memory_reserved,
                    "memory_reserved_gb": memory_reserved / (1024**3),
                    "memory_free_gb": (device_props.total_memory - memory_reserved) / (1024**3),
                })
            
            info["devices"].append(device_info)
    
    return info


def optimize_gpu_memory() -> None:
    """Optimize GPU memory usage with various techniques.
    
    This function applies several optimizations to reduce GPU memory usage
    and improve training stability.
    """
    if not torch.cuda.is_available():
        logger.info("CUDA not available, skipping GPU optimization")
        return
    
    logger.info("Optimizing GPU memory usage...")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Set memory allocation strategy
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Enable memory-efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        logger.info("Enabled Flash Attention for memory efficiency")
    except AttributeError:
        pass  # Flash attention not available in this PyTorch version
    
    # Set cudnn benchmarking for consistent input sizes
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    logger.info("GPU memory optimization completed")


def monitor_gpu_memory() -> Dict[str, float]:
    """Monitor current GPU memory usage.
    
    Returns:
        Dictionary with memory statistics in GB
        
    Example:
        >>> memory_stats = monitor_gpu_memory()
        >>> print(f"GPU memory used: {memory_stats['allocated_gb']:.2f} GB")
    """
    if not torch.cuda.is_available():
        return {}
    
    device = torch.cuda.current_device()
    
    # Get memory statistics
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    
    stats = {
        "device": device,
        "allocated_bytes": allocated,
        "allocated_gb": allocated / (1024**3),
        "reserved_bytes": reserved,
        "reserved_gb": reserved / (1024**3),
        "total_bytes": total,
        "total_gb": total / (1024**3),
        "free_gb": (total - reserved) / (1024**3),
        "utilization_percent": (reserved / total) * 100,
    }
    
    return stats


def log_gpu_memory(logger: logging.Logger, prefix: str = "") -> None:
    """Log current GPU memory usage.
    
    Args:
        logger: Logger instance
        prefix: Optional prefix for log message
    """
    if not torch.cuda.is_available():
        return
    
    stats = monitor_gpu_memory()
    if stats:
        message = (f"{prefix}GPU Memory - "
                  f"Allocated: {stats['allocated_gb']:.2f} GB, "
                  f"Reserved: {stats['reserved_gb']:.2f} GB, "
                  f"Free: {stats['free_gb']:.2f} GB "
                  f"({stats['utilization_percent']:.1f}% used)")
        logger.info(message)


def cleanup_gpu_memory() -> None:
    """Clean up GPU memory by clearing cache and collecting garbage.
    
    This function should be called periodically during training to prevent
    memory leaks and fragmentation.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    gc.collect()


def check_gpu_requirements(required_memory_gb: float = 8.0) -> bool:
    """Check if GPU meets minimum requirements.
    
    Args:
        required_memory_gb: Minimum required GPU memory in GB
        
    Returns:
        True if requirements are met
        
    Example:
        >>> if check_gpu_requirements(16.0):
        ...     print("GPU suitable for large model training")
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available")
        return False
    
    gpu_info = get_gpu_info()
    
    for device in gpu_info["devices"]:
        if device["total_memory_gb"] >= required_memory_gb:
            logger.info(f"GPU {device['id']} ({device['name']}) meets requirements: "
                       f"{device['total_memory_gb']:.1f} GB >= {required_memory_gb} GB")
            return True
    
    max_memory = max(device["total_memory_gb"] for device in gpu_info["devices"])
    logger.warning(f"No GPU meets memory requirements. "
                  f"Required: {required_memory_gb} GB, "
                  f"Available: {max_memory:.1f} GB")
    return False


def set_optimal_cuda_device(memory_threshold_gb: float = 1.0) -> Optional[int]:
    """Select the optimal CUDA device based on available memory.
    
    Args:
        memory_threshold_gb: Minimum free memory required in GB
        
    Returns:
        Device ID of optimal GPU, or None if no suitable GPU found
        
    Example:
        >>> device_id = set_optimal_cuda_device(4.0)
        >>> if device_id is not None:
        ...     torch.cuda.set_device(device_id)
    """
    if not torch.cuda.is_available():
        return None
    
    gpu_info = get_gpu_info()
    
    # Find GPU with most free memory
    best_device = None
    max_free_memory = 0
    
    for device in gpu_info["devices"]:
        free_memory = device.get("memory_free_gb", device["total_memory_gb"])
        
        if free_memory >= memory_threshold_gb and free_memory > max_free_memory:
            best_device = device["id"]
            max_free_memory = free_memory
    
    if best_device is not None:
        torch.cuda.set_device(best_device)
        logger.info(f"Selected GPU {best_device} with {max_free_memory:.1f} GB free memory")
        return best_device
    else:
        logger.warning(f"No GPU found with at least {memory_threshold_gb} GB free memory")
        return None


def estimate_model_memory(
    num_parameters: int,
    batch_size: int = 1,
    sequence_length: int = 512,
    dtype_bytes: int = 2,  # bfloat16/float16
    gradient_multiplier: float = 2.0,  # gradients + optimizer states
    activation_multiplier: float = 4.0,  # activation memory
) -> float:
    """Estimate GPU memory requirements for model training.
    
    Args:
        num_parameters: Number of model parameters
        batch_size: Training batch size
        sequence_length: Input sequence length
        dtype_bytes: Bytes per parameter (2 for fp16/bf16, 4 for fp32)
        gradient_multiplier: Multiplier for gradients and optimizer states
        activation_multiplier: Multiplier for activation memory
        
    Returns:
        Estimated memory usage in GB
        
    Example:
        >>> memory_gb = estimate_model_memory(
        ...     num_parameters=125_000_000,  # 125M parameters
        ...     batch_size=8,
        ...     sequence_length=512
        ... )
        >>> print(f"Estimated memory: {memory_gb:.2f} GB")
    """
    # Model weights
    model_memory = num_parameters * dtype_bytes
    
    # Gradients and optimizer states (Adam uses ~2x model size)
    gradient_memory = model_memory * gradient_multiplier
    
    # Activation memory (depends on batch size and sequence length)
    # Rough estimate: batch_size * sequence_length * hidden_size * layers * dtype_bytes
    # Simplified estimation based on parameters
    activation_memory = (batch_size * sequence_length * 
                        (num_parameters ** 0.5) * dtype_bytes * activation_multiplier)
    
    total_memory_bytes = model_memory + gradient_memory + activation_memory
    total_memory_gb = total_memory_bytes / (1024**3)
    
    logger.info(f"Memory estimation:")
    logger.info(f"  Model weights: {model_memory / (1024**3):.2f} GB")
    logger.info(f"  Gradients/optimizer: {gradient_memory / (1024**3):.2f} GB")
    logger.info(f"  Activations: {activation_memory / (1024**3):.2f} GB")
    logger.info(f"  Total estimated: {total_memory_gb:.2f} GB")
    
    return total_memory_gb


class GPUMemoryTracker:
    """Context manager for tracking GPU memory usage."""
    
    def __init__(self, logger: logging.Logger, description: str = "Operation"):
        """Initialize memory tracker.
        
        Args:
            logger: Logger instance
            description: Description of the operation being tracked
        """
        self.logger = logger
        self.description = description
        self.start_stats = None
        self.end_stats = None
    
    def __enter__(self):
        """Enter context and record initial memory state."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_stats = monitor_gpu_memory()
            self.logger.info(f"{self.description} - Start memory: "
                           f"{self.start_stats.get('allocated_gb', 0):.2f} GB allocated")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and report memory usage."""
        if torch.cuda.is_available() and self.start_stats:
            torch.cuda.synchronize()
            self.end_stats = monitor_gpu_memory()
            
            memory_diff = (self.end_stats.get('allocated_gb', 0) - 
                          self.start_stats.get('allocated_gb', 0))
            
            self.logger.info(f"{self.description} - End memory: "
                           f"{self.end_stats.get('allocated_gb', 0):.2f} GB allocated "
                           f"(Δ{memory_diff:+.2f} GB)")
    
    def peak_memory(self) -> float:
        """Get peak memory usage during the tracked operation.
        
        Returns:
            Peak memory usage in GB
        """
        if torch.cuda.is_available():
            peak_memory_bytes = torch.cuda.max_memory_allocated()
            return peak_memory_bytes / (1024**3)
        return 0.0