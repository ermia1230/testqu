#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CIRCUITS="${CIRCUITS:-GHZ QFT QAOA}"
QUBITS="${QUBITS:-10 15 20 25 30}"
BACKENDS="${BACKENDS:-nvidia tensornet tensornet-mps}"
SHOTS="${SHOTS:-10 100 1024}"
REPETITIONS="${REPETITIONS:-10}"
WARMUP="${WARMUP:-1}"
OUT="${OUT:-results/python_cpp_comparison.csv}"
SEED="${SEED:-1234}"

: "${CUDAQ_MPS_MAX_BOND:=64}"
: "${CUDAQ_MPS_ABS_CUTOFF:=1e-5}"
: "${CUDAQ_MPS_RELATIVE_CUTOFF:=1e-5}"
: "${CUDAQ_MPS_SVD_ALGO:=gesvd}"
export CUDAQ_MPS_MAX_BOND
export CUDAQ_MPS_ABS_CUTOFF
export CUDAQ_MPS_RELATIVE_CUTOFF
export CUDAQ_MPS_SVD_ALGO

mkdir -p "$(dirname "$OUT")"

for backend in $BACKENDS; do
  make -C cpp "TARGET_BACKEND=$backend"
  cpp_exe="cpp/build/cudaq-benchmark-$backend"

  for circuit in $CIRCUITS; do
    for qubits in $QUBITS; do
      for shots in $SHOTS; do
        python3 scripts/compare_python_cpp.py \
          --no-build \
          --cpp-exe "$cpp_exe" \
          --backend "$backend" \
          --circuit "$circuit" \
          --num-qubits "$qubits" \
          --shots "$shots" \
          --repetitions "$REPETITIONS" \
          --warmup "$WARMUP" \
          --seed "$SEED" \
          --output-csv "$OUT"
      done
    done
  done
done

echo "Wrote $OUT"
