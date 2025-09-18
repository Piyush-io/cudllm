import os
import argparse
from src.core.fsr_framework import FSR_Framework
from src.core.benchmarks import vector_add_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FSR search for CUDA kernel generation")
    parser.add_argument("--arch", default=os.environ.get("GPU_ARCH", "sm_80"), help="Target GPU arch, e.g. sm_80, sm_86, sm_90")
    parser.add_argument("--depth", type=int, default=int(os.environ.get("FSR_DEPTH", 2)), help="Search max depth")
    parser.add_argument("--candidates", type=int, default=int(os.environ.get("FSR_CANDIDATES", 2)), help="Candidates per round")
    parser.add_argument("--dry-run", action="store_true", help="Do not compile/validate/profile; only generate candidates and save them")
    parser.add_argument("--offline", action="store_true", help="Skip LLM call and write placeholder kernels for smoke-testing")
    args = parser.parse_args()

    task = vector_add_task()

    fsr = FSR_Framework(max_depth=args.depth, candidates_per_round=args.candidates)

    if args.dry_run:
        prompt = fsr.prompts.create_initial_prompt(task.description, task.host_code, {"arch": args.arch}, fsr.N)
        if args.offline:
            kernels = [
                "#include <cstdio>\nint main(){ printf(\"OK\\n\"); printf(\"TIME_MS 0.0\\n\"); return 0;}"
                for _ in range(args.candidates)
            ]
        else:
            kernels = fsr.generate_kernels(prompt, args.candidates)
        print(f"generated_candidates={len(kernels)}")
        for i, k in enumerate(kernels):
            out = f"candidate_{i}.cu"
            with open(out, "w") as f:
                f.write(k)
            print(f"saved: {out}")
        return

    result = fsr.fsr_search(task.description, task.host_code, {"arch": args.arch})

    print(f"iterations={result.iterations}")
    print(f"best_time_ms={result.best_time_ms}")
    print(f"num_candidates={len(result.candidates)}")

    if result.best_kernel:
        with open("best_kernel.cu", "w") as f:
            f.write(result.best_kernel)
        print("saved: best_kernel.cu")
    else:
        print("no valid kernel found")


if __name__ == "__main__":
    main()
