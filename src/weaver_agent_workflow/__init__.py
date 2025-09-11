from .llm.interface import LLMInterface, LLMConfig
from .fsr.framework import FSRFramework
from .benchmarks.tasks import BenchmarkSuite
import os
import sys


def get_hardware_spec() -> dict:
    """Detect hardware specifications for CUDA kernels."""
    hw_spec = {
        "name": "Generic GPU",
        "architecture": "Unknown",
        "compute_capability": "8.0",
        "memory_gb": "Unknown",
        "memory_bandwidth_gb_s": "Unknown",
        "cuda_cores": "Unknown",
        "sm_count": "Unknown",
        "max_threads_per_block": 1024,
        "max_shared_memory_per_block": 49152,
        "warp_size": 32
    }
    
    try:
        import cupy as cp
        device = cp.cuda.Device(0)
        with device:
            props = cp.cuda.runtime.getDeviceProperties(0)
            hw_spec.update({
                "name": props["name"].decode("utf-8"),
                "compute_capability": f"{props['major']}.{props['minor']}",
                "memory_gb": props["totalGlobalMem"] // (1024**3),
                "sm_count": props["multiProcessorCount"],
                "max_threads_per_block": props["maxThreadsPerBlock"],
                "max_shared_memory_per_block": props["sharedMemPerBlock"],
                "warp_size": props["warpSize"]
            })
    except (ImportError, Exception):
        pass
    
    return hw_spec


def main() -> None:
    """Main entry point for the weaver-agent-workflow."""
    try:
        # Check for OpenAI API key
        if not os.environ.get("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY environment variable not set")
            print("Set it with: export OPENAI_API_KEY=your_api_key")
            return
        
        # Run benchmark task
        task = BenchmarkSuite.get_identity_task()
        print(f"Running task: {task.name}")
        print(f"Description: {task.description}")

        # Initialize components
        llm_config = LLMConfig(model="gpt-4o-2024-08-06", temperature=0.1)
        llm = LLMInterface(llm_config)
        framework = FSRFramework(
            llm_model=llm, 
            max_depth=3, 
            candidates_per_round=3,
            reference_implementation=task.reference_implementation
        )

        # Get hardware specs
        hw_spec = get_hardware_spec()
        print(f"Detected GPU: {hw_spec['name']}")
        print(f"Compute Capability: {hw_spec['compute_capability']}")

        prompt = llm.prompt.create_initial_prompt(task.description, task.host_code, hw_spec=hw_spec)
        test_cases = task.generate_test_cases()
        
        print("Starting FSR search...")
        best = framework.fsr_search(prompt, test_cases)
        
        if best is None:
            print("No valid kernel found. This is expected without proper CUDA toolchain setup.")
            print("Make sure you have:")
            print("  1. NVIDIA GPU with CUDA drivers")
            print("  2. CUDA toolkit installed") 
            print("  3. CuPy installed for your CUDA version")
            print("  4. nvcc compiler in PATH")
        else:
            print(f"✅ Best kernel found: {best.name}")
            if best.metadata:
                exec_time = best.metadata.get('execution_time', 'unknown')
                print(f"   Performance: {exec_time} ms")
                optimizations = best.metadata.get('optimizations', [])
                if optimizations:
                    print(f"   Optimizations: {', '.join(optimizations)}")
                    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
