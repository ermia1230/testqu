#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Common CUDA-Q installation bin directories (searched when nvq++ is not on PATH).
_CUDAQ_BIN_CANDIDATES = [
    Path("/content/.cudaq/bin"),
    Path("/opt/nvidia/cudaq/bin"),
    Path.home() / ".cudaq" / "bin",
]


def _resolve_env() -> dict:
    """Return an env dict with the CUDA-Q bin directory prepended to PATH."""
    env = os.environ.copy()
    if shutil.which("nvq++") is not None:
        return env
    for candidate in _CUDAQ_BIN_CANDIDATES:
        if (candidate / "nvq++").exists():
            env["PATH"] = str(candidate) + os.pathsep + env.get("PATH", "")
            return env
    return env


def split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def split_ints(value):
    return [int(item) for item in split_csv(value)]


def run(command, env=None):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def write_summary(output_csv):
    path = REPO_ROOT / output_csv
    if not path.exists():
        print(f"No comparison CSV found at {path}")
        return

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"No rows found in {path}")
        return

    print("\nQuick comparison summary")
    print("backend,circuit,qubits,shots,python_sample,cpp_sample,speedup")
    for row in rows[-10:]:
        print(
            ",".join(
                [
                    row["backend"],
                    row["circuit"],
                    row["qubits"],
                    row["shots"],
                    row["python_sample_time"],
                    row["cpp_sample_time"],
                    row["speedup_cpp_over_python"],
                ]
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run a short Colab-friendly Python vs C++ CUDA-Q comparison."
    )
    parser.add_argument("--backend", default="nvidia")
    parser.add_argument("--circuits", default="GHZ,QFT,QAOA")
    parser.add_argument("--qubits", default="10,15")
    parser.add_argument("--shots", default="100")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output-csv", default="results/colab_quick_python_cpp.csv"
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python interpreter with cudaq installed (default: sys.executable).",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Use an already-built C++ executable.",
    )
    args = parser.parse_args()

    env = _resolve_env()

    if shutil.which("nvq++", path=env.get("PATH")) is None:
        raise SystemExit(
            "nvq++ was not found. Install the CUDA-Q C++ pre-built binaries "
            "and source set_env.sh before running this script.\n"
            f"Searched PATH: {env.get('PATH')}"
        )

    circuits = split_csv(args.circuits)
    qubits = split_ints(args.qubits)
    shots_values = split_ints(args.shots)

    if not args.skip_build:
        run(["make", "-C", "cpp", f"TARGET_BACKEND={args.backend}"], env=env)

    cpp_exe = REPO_ROOT / "cpp" / "build" / f"cudaq-benchmark-{args.backend}"

    for circuit in circuits:
        for n_qubits in qubits:
            for shots in shots_values:
                run(
                    [
                        sys.executable,
                        "scripts/compare_python_cpp.py",
                        "--no-build",
                        "--python-exe",
                        args.python_exe,
                        "--cpp-exe",
                        str(cpp_exe),
                        "--backend",
                        args.backend,
                        "--circuit",
                        circuit,
                        "--num-qubits",
                        str(n_qubits),
                        "--shots",
                        str(shots),
                        "--repetitions",
                        str(args.repetitions),
                        "--warmup",
                        str(args.warmup),
                        "--seed",
                        str(args.seed),
                        "--output-csv",
                        args.output_csv,
                    ],
                    env=env,
                )

    write_summary(args.output_csv)


if __name__ == "__main__":
    main()
