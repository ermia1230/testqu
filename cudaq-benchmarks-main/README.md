# Benchmark suite for MPS/Tensor Network simulation
This repository proposes a set of five configurable quantum circuits for evaluation
of Nvidia's CUDA-Q simulator backends, targeting specifically plain Tensor Network
and Matrix Product State (MPS) simulators.

## Usage
To obtain help on the utilization of the benchmark script, use:
```
python benchmark.py --help
```

Depending on the architecture of the executing system - and notably on Arm platform -,
it might be simpler to execute CUDA-Q within a Docker container. Nvidia provides a
[Docker image](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum)
which streamlines this process. We propose an experimental script `scripts/run.sh`, which
automatizes creating and running the container, used as a wrapper:
```
scripts/./run.sh python3 benchmark.py --help
```

## Python vs C++ CUDA-Q comparison

This repository also contains a small C++ CUDA-Q benchmark under `cpp/` for
checking whether Python frontend/kernel overhead is material for these circuits.
The goal is measurement, not assuming C++ is faster. If most runtime is inside
`cudaq.sample` and the selected simulator backend, Python and C++ can be similar.

Implemented C++ circuits:

- `GHZ`
- `QFT`
- `QAOA`

`QuantumVolume` is not implemented in C++ yet because the Python version relies
on per-layer random SU(4) custom operations; that needs a careful CUDA-Q C++
custom-operation implementation to keep the circuit equivalent.

### Build the C++ benchmark

CUDA-Q C++ simulator targets are selected when compiling with `nvq++ --target`.
The Makefile builds one executable per backend:

```
make -C cpp TARGET_BACKEND=nvidia
make -C cpp TARGET_BACKEND=tensornet
make -C cpp TARGET_BACKEND=tensornet-mps
```

The generated executable path is:

```
cpp/build/cudaq-benchmark-<backend>
```

### Run one C++ benchmark

```
cpp/build/cudaq-benchmark-nvidia \
  --circuit GHZ \
  --num-qubits 20 \
  --num-shots 1024 \
  --repetitions 10 \
  --warmup 1 \
  --target nvidia \
  --csv results/cpp.csv
```

The C++ `--target` argument is recorded in the output CSV. The actual CUDA-Q C++
backend is the one used at build time via `TARGET_BACKEND`.

The CSV row reports sample-only timing and per-iteration total timing:

- `sample_*_seconds`: timing around `cudaq::sample(...)`
- `total_*_seconds`: timing around the full per-iteration call path
- `program_total_seconds`: complete process runtime for that configuration

### Run a Python vs C++ comparison

```
python3 scripts/compare_python_cpp.py \
  --backend nvidia \
  --circuit GHZ \
  --num-qubits 20 \
  --shots 1024 \
  --repetitions 10 \
  --warmup 1 \
  --output-csv results/python_cpp_comparison.csv
```

The script runs `benchmark.py`, runs the matching C++ executable, and appends a
row with:

- `python_total_time`
- `python_sample_time`
- `cpp_total_time`
- `cpp_sample_time`
- `speedup_cpp_over_python`

`speedup_cpp_over_python` is computed from sample mean time when both sides
provide it. Values above 1 mean the C++ run was faster for that measurement.
For `QFT` and `QAOA`, the comparison script generates the random input state or
weights once and passes the exact same values to both frontends.

### Reproducible sweep

```
scripts/run_python_cpp_comparison.sh
```

By default this sweeps:

- circuits: `GHZ QFT QAOA`
- qubits: `10 15 20 25 30`
- backends: `nvidia tensornet tensornet-mps`
- shots: `10 100 1024`
- repetitions: `10`

You can override any sweep dimension:

```
CIRCUITS="GHZ QAOA" QUBITS="10 15 20" BACKENDS="tensornet-mps" \
  scripts/run_python_cpp_comparison.sh
```

For MPS runs, the script exports the same environment variables for both Python
and C++:

```
CUDAQ_MPS_MAX_BOND
CUDAQ_MPS_ABS_CUTOFF
CUDAQ_MPS_RELATIVE_CUTOFF
CUDAQ_MPS_SVD_ALGO
```

Set these before invoking the script to override the defaults.

### Interpreting results

Use `python_sample_time` and `cpp_sample_time` to test the narrow question:
whether the language frontend changes the measured `sample` call. Use total
times to include more host-side benchmark overhead. Similar sample times suggest
the bottleneck is probably inside CUDA-Q or the simulator backend rather than in
Python circuit construction.

## Citation
G. Schieffer, S. Markidis, and I. Peng. (2025). Harnessing CUDA-Q's MPS for Tensor Network Simulations of Large-Scale Quantum Circuits. *2025 33rd Euromicro International Conference on Parallel, Distributed and Network-Based Processing*.     

Arxiv pre-print: https://doi.org/10.48550/arXiv.2501.15939
