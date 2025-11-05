"""Interactive FSR runner with simple menu interface."""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from src.core.benchmarks import vector_add_task
from src.core.fsr_framework import FSR_Framework


def detect_gpu_arch() -> Optional[str]:
    """Auto-detect GPU architecture using nvidia-smi."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        line = proc.stdout.strip().splitlines()[0].strip()
        if not line or "." not in line:
            return None
        major, minor = line.split(".", 1)
        return f"sm_{int(major)}{int(minor)}"
    except Exception:
        return None


def get_input(prompt: str, default: str = "") -> str:
    """Get input with default value."""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"{prompt}: ").strip()


def yes_no(prompt: str, default: bool = False) -> bool:
    """Ask yes/no question."""
    default_str = "Y/n" if default else "y/N"
    response = input(f"{prompt} [{default_str}]: ").strip().lower()
    if not response:
        return default
    return response in ["y", "yes"]


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )


def main():
    """Interactive FSR menu."""
    print("\n" + "=" * 60)
    print("FSR CUDA Kernel Generator - Interactive Mode")
    print("=" * 60 + "\n")

    # Environment check
    print("Environment Setup\n")

    # ChromaDB Cloud credentials check
    chroma_mode = os.environ.get("CHROMA_MODE", "cloud")
    chroma_api_key = os.environ.get("CHROMA_API_KEY", "")
    chroma_tenant = os.environ.get("CHROMA_TENANT", "")
    chroma_database = os.environ.get("CHROMA_DATABASE", "")

    if chroma_mode.lower() == "cloud":
        if not all([chroma_api_key, chroma_tenant, chroma_database]):
            print("WARNING: ChromaDB Cloud credentials not found in environment!")
            print("Please set the following environment variables:")
            print("  - CHROMA_API_KEY")
            print("  - CHROMA_TENANT")
            print("  - CHROMA_DATABASE")
            print("\nOr create a .env file with these values.")
            print("RAG retrieval will fail without proper credentials.\n")

            if not yes_no("Continue anyway?", default=False):
                print("\nCancelled. Please configure ChromaDB credentials first.\n")
                return
        else:
            print(f"ChromaDB Cloud configured: {chroma_tenant}/{chroma_database}")
    else:
        persist_path = os.environ.get("CHROMA_PERSIST_PATH", "./chroma_db")
        print(f"ChromaDB Persistent mode: {persist_path}")

    print()

    # GPU Architecture
    detected_arch = detect_gpu_arch()
    if detected_arch:
        print(f"Auto-detected GPU: {detected_arch}")
        arch = get_input("GPU architecture", detected_arch)
    else:
        print("WARNING: Could not auto-detect GPU")
        arch = get_input("GPU architecture (e.g., sm_80, sm_86, sm_90)", "sm_80")

    # Search parameters
    print("\nSearch Configuration\n")
    depth = int(get_input("Maximum search depth (iterations)", "2"))
    candidates = int(get_input("Candidates per iteration", "2"))

    # Logging
    log_level = get_input("Log level (DEBUG/INFO/WARNING)", "INFO")
    setup_logging(log_level)

    # Dry run option
    print()
    dry_run = yes_no("Dry run? (generate only, no compile/run)", default=False)

    # Confirmation
    print("\n" + "=" * 60)
    print("Configuration Summary:")
    print("=" * 60)
    print(f"  GPU Architecture: {arch}")
    print(f"  Search Depth: {depth}")
    print(f"  Candidates/Round: {candidates}")
    print(f"  Log Level: {log_level}")
    print(f"  Dry Run: {dry_run}")
    print(f"  ChromaDB Mode: {chroma_mode}")
    print("=" * 60 + "\n")

    if not yes_no("Start FSR search?", default=True):
        print("\nCancelled.\n")
        return

    # Run FSR
    print("\nStarting FSR search...\n")

    task = vector_add_task()
    fsr = FSR_Framework(max_depth=depth, candidates_per_round=candidates)

    if dry_run:
        print("Generating candidates without compilation/execution...\n")
        prompt = fsr.prompts.create_initial_prompt(
            task.description, task.host_code, {"arch": arch}, candidates
        )
        kernels = fsr.generate_kernels(prompt, candidates)
        print(f"\nGenerated {len(kernels)} candidates\n")

        for i, kernel in enumerate(kernels):
            out_path = Path(f"candidate_{i}.cu")
            out_path.write_text(kernel)
            print(f"  Saved: {out_path}")

        print("\nDry run complete.\n")
        return

    # Full search
    result = fsr.fsr_search(task.description, task.host_code, {"arch": arch})

    # Results
    print("\n" + "=" * 60)
    print("FSR Search Results")
    print("=" * 60)
    print(f"  Iterations: {result.iterations}")
    print(f"  Total Candidates: {len(result.candidates)}")
    if result.best_time_ms > 0:
        print(f"  Best Time: {result.best_time_ms:.3f} ms")
    else:
        print("  Best Time: N/A (no valid kernel)")
    print("=" * 60 + "\n")

    if result.best_kernel:
        best_path = Path("best_kernel.cu")
        best_path.write_text(result.best_kernel)
        print(f"Best kernel saved: {best_path}\n")
    else:
        print("WARNING: No valid kernel found\n")

    # Cleanup
    try:
        fsr.compilation_verifier.cleanup()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.\n")
    except Exception as e:
        print(f"\n\nERROR: {e}\n")
        raise
