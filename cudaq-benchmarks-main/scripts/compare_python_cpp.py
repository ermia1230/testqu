#!/usr/bin/env python3
import argparse
import csv
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(command, env=None):
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        print(completed.stdout, end="", file=sys.stderr)
        print(completed.stderr, end="", file=sys.stderr)
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed


def read_last_row(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"no rows found in {path}")
    return rows[-1]


def as_float(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


def write_comparison_row(path, row):
    fieldnames = [
        "backend",
        "circuit",
        "qubits",
        "shots",
        "repetitions",
        "warmup",
        "python_total_time",
        "python_sample_time",
        "cpp_total_time",
        "cpp_sample_time",
        "speedup_cpp_over_python",
        "speedup_basis",
    ]

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists()
    with output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def default_cpp_exe(backend):
    return REPO_ROOT / "cpp" / "build" / f"cudaq-benchmark-{backend}"


def shared_parameter_args(circuit, qubits, seed):
    rng = random.Random(seed)
    if circuit == "QFT":
        bits = "".join(str(rng.randrange(2)) for _ in range(qubits))
        return ["--qft-input-state", bits]
    if circuit == "QAOA":
        weights = [str(-1 if rng.randrange(2) == 0 else 1)
                   for _ in range(qubits * (qubits - 1) // 2)]
        return ["--qaoa-weights", ",".join(weights)]
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Run the Python and C++ CUDA-Q benchmarks with matching settings."
    )
    parser.add_argument("--circuit", required=True, choices=["GHZ", "QFT", "QAOA"])
    parser.add_argument("-n", "--num-qubits", type=int, required=True)
    parser.add_argument("-s", "--shots", "--num-shots", type=int, default=1024)
    parser.add_argument("-i", "--repetitions", "--iter", type=int, default=10)
    parser.add_argument("-w", "--warmup", type=int, default=1)
    parser.add_argument("--backend", "--target", dest="backend", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-csv", default="results/python_cpp_comparison.csv")
    parser.add_argument("--cpp-exe", type=Path)
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip building the C++ executable before running it.",
    )
    args = parser.parse_args()

    cpp_exe = args.cpp_exe or default_cpp_exe(args.backend)
    if not cpp_exe.is_absolute():
        cpp_exe = REPO_ROOT / cpp_exe

    if not args.no_build:
        run_command(["make", "-C", "cpp", f"TARGET_BACKEND={args.backend}"])

    if not cpp_exe.exists():
        raise FileNotFoundError(
            f"C++ executable not found: {cpp_exe}. Build with "
            f"`make -C cpp TARGET_BACKEND={args.backend}` or pass --cpp-exe."
        )

    env = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="cudaq-bench-compare-") as tmp:
        tmp_path = Path(tmp)
        python_csv = tmp_path / "python.csv"
        cpp_csv = tmp_path / "cpp.csv"

        python_cmd = [
            sys.executable,
            "benchmark.py",
            "--circuit",
            args.circuit,
            "-n",
            str(args.num_qubits),
            "-s",
            str(args.shots),
            "-i",
            str(args.repetitions),
            "-w",
            str(args.warmup),
            "--target",
            args.backend,
            "--csv",
            str(python_csv),
        ]
        cpp_cmd = [
            str(cpp_exe),
            "--circuit",
            args.circuit,
            "-n",
            str(args.num_qubits),
            "-s",
            str(args.shots),
            "-i",
            str(args.repetitions),
            "-w",
            str(args.warmup),
            "--target",
            args.backend,
            "--csv",
            str(cpp_csv),
        ]

        if args.seed is not None:
            python_cmd.extend(["--seed", str(args.seed)])
            cpp_cmd.extend(["--seed", str(args.seed)])

        parameter_args = shared_parameter_args(args.circuit, args.num_qubits, args.seed)
        python_cmd.extend(parameter_args)
        cpp_cmd.extend(parameter_args)

        print("Running Python benchmark:", " ".join(python_cmd), file=sys.stderr)
        run_command(python_cmd, env=env)
        print("Running C++ benchmark:", " ".join(cpp_cmd), file=sys.stderr)
        run_command(cpp_cmd, env=env)

        py_row = read_last_row(python_csv)
        cpp_row = read_last_row(cpp_csv)

    python_sample = as_float(py_row, "sample_mean_seconds")
    cpp_sample = as_float(cpp_row, "sample_mean_seconds")
    python_total = as_float(py_row, "total_mean_seconds")
    cpp_total = as_float(cpp_row, "total_mean_seconds")

    if python_sample is not None and cpp_sample not in (None, 0.0):
        speedup = python_sample / cpp_sample
        basis = "sample_mean_seconds"
    elif python_total is not None and cpp_total not in (None, 0.0):
        speedup = python_total / cpp_total
        basis = "total_mean_seconds"
    else:
        speedup = ""
        basis = ""

    comparison = {
        "backend": args.backend,
        "circuit": args.circuit,
        "qubits": args.num_qubits,
        "shots": args.shots,
        "repetitions": args.repetitions,
        "warmup": args.warmup,
        "python_total_time": python_total,
        "python_sample_time": python_sample,
        "cpp_total_time": cpp_total,
        "cpp_sample_time": cpp_sample,
        "speedup_cpp_over_python": speedup,
        "speedup_basis": basis,
    }

    write_comparison_row(args.output_csv, comparison)
    print(
        f"{args.backend} {args.circuit} n={args.num_qubits} shots={args.shots}: "
        f"speedup={speedup} ({basis})"
    )


if __name__ == "__main__":
    main()
