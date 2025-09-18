import os
import argparse
import logging
import subprocess
from src.core.fsr_framework import FSR_Framework
from src.core.benchmarks import vector_add_task


def _detect_sm_arch() -> str | None:
    """Detect GPU SM arch (e.g., sm_75) using nvidia-smi compute capability."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        line = proc.stdout.strip().splitlines()[0].strip()
        # Expected like "7.5" or "8.6"
        if not line or "." not in line:
            return None
        major_str, minor_str = line.split(".", 1)
        major = int(major_str)
        minor = int(minor_str)
        return f"sm_{major}{minor}"
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FSR search for CUDA kernel generation")
    parser.add_argument(
        "--arch",
        default=os.environ.get("GPU_ARCH", "auto"),
        help="Target GPU arch, e.g. sm_80, sm_86, sm_90, or 'auto'",
    )
    parser.add_argument("--depth", type=int, default=int(os.environ.get("FSR_DEPTH", 2)), help="Search max depth")
    parser.add_argument("--candidates", type=int, default=int(os.environ.get("FSR_CANDIDATES", 2)), help="Candidates per round")
    parser.add_argument("--dry-run", action="store_true", help="Do not compile/validate/profile; only generate candidates and save them")
    parser.add_argument("--offline", action="store_true", help="Skip LLM call and write placeholder kernels for smoke-testing")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"), help="Logging level: DEBUG, INFO, WARNING")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("run_fsr")

    task = vector_add_task()

    fsr = FSR_Framework(max_depth=args.depth, candidates_per_round=args.candidates)

    # Resolve architecture
    resolved_arch = args.arch
    if resolved_arch.lower() in ("auto", "detect", "sm_auto", "auto_detect"):
        detected = _detect_sm_arch()
        if detected:
            resolved_arch = detected
            logging.getLogger("run_fsr").info("auto-detected GPU arch: %s", resolved_arch)
        else:
            resolved_arch = "sm_80"
            logging.getLogger("run_fsr").warning(
                "failed to auto-detect GPU arch; falling back to %s", resolved_arch
            )

    if args.dry_run:
        prompt = fsr.prompts.create_initial_prompt(task.description, task.host_code, {"arch": resolved_arch}, fsr.N)
        if args.offline:
            kernels = [
                "#include <cstdio>\nint main(){ printf(\"OK\\n\"); printf(\"TIME_MS 0.0\\n\"); return 0;}"
                for _ in range(args.candidates)
            ]
        else:
            kernels = fsr.generate_kernels(prompt, args.candidates)
        log.info("generated_candidates=%d", len(kernels))
        for i, k in enumerate(kernels):
            out = f"candidate_{i}.cu"
            with open(out, "w") as f:
                f.write(k)
            log.info("saved: %s", out)
        return

    result = fsr.fsr_search(task.description, task.host_code, {"arch": resolved_arch})

    log.info("iterations=%d", result.iterations)
    log.info("best_time_ms=%.3f", result.best_time_ms)
    log.info("num_candidates=%d", len(result.candidates))

    if result.best_kernel:
        with open("best_kernel.cu", "w") as f:
            f.write(result.best_kernel)
        log.info("saved: best_kernel.cu")
    else:
        log.warning("no valid kernel found")


if __name__ == "__main__":
    main()
