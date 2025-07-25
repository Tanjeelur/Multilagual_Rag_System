import os
import sys
import subprocess
import time
import platform

def check_gpu_availability():
    """Check if GPU is available for PyTorch"""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("No GPU available, using CPU only")
            return False
    except ImportError:
        print("PyTorch not installed, cannot check GPU availability")
        return False

def clear_cache():
    """Clear the database cache before starting the API"""
    print("Clearing database cache...")
    try:
        subprocess.run(["python", "multilingual-rag/clear_cache.py"], check=True)
        print("Database cache cleared successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error clearing cache: {e}")
        sys.exit(1)

def run_api():
    """Run the API with optimized settings"""
    print("Starting API server with optimized settings...")
    try:
        # Change to the multilingual-rag directory
        os.chdir("multilingual-rag")
        
        # Set environment variables to optimize memory usage
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads
        env["MKL_NUM_THREADS"] = "1"  # Limit MKL threads
        env["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism
        
        # Set PyTorch memory management options
        env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"  # Disable CUDA memory caching
        
        # Check if we should use GPU
        use_gpu = check_gpu_availability()
        if use_gpu:
            # If GPU is available, set environment variables for GPU optimization
            env["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the first GPU
            print("Using GPU for model inference")
        else:
            # If no GPU, optimize for CPU usage
            env["CUDA_VISIBLE_DEVICES"] = ""  # Hide CUDA devices
            # Additional CPU optimizations
            env["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TensorFlow logging
            print("Optimizing for CPU-only operation")
        
        # Set memory limits based on platform
        if platform.system() == "Windows":
            # Windows-specific optimizations
            print("Applying Windows-specific optimizations")
            # No specific Windows memory limit setting, rely on environment variables
        else:
            # Linux/Mac optimizations
            print("Applying Linux/Mac optimizations")
            # Could add ulimit settings here for Linux/Mac
        
        # Run the API server with reduced worker count
        subprocess.run(
            ["python", "-m", "uvicorn", "src.api:app", "--reload", "--workers", "1", "--limit-concurrency", "5"], 
            env=env,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running API: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("API server stopped by user")

def cleanup_memory():
    """Perform memory cleanup before starting the API"""
    print("Performing memory cleanup...")
    try:
        # Try to clean up Python memory
        import gc
        gc.collect()
        
        # If PyTorch is available, clean up CUDA memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA memory cache cleared")
        except ImportError:
            pass
        
        # On Windows, try to run the garbage collector more aggressively
        if platform.system() == "Windows":
            for _ in range(3):  # Run multiple times for better cleanup
                gc.collect()
                time.sleep(0.5)
        
        print("Memory cleanup completed")
    except Exception as e:
        print(f"Warning: Memory cleanup encountered an issue: {e}")

if __name__ == "__main__":
    print("Starting optimized RAG API with memory management...")
    
    # Perform initial memory cleanup
    cleanup_memory()
    
    # Clear the database cache
    clear_cache()
    
    # Wait a moment for the cache clearing to complete
    time.sleep(1)
    
    # Run the API with optimizations
    run_api()