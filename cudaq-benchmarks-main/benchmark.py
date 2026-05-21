#!/usr/bin/env python3
import time
import argparse
import sys
import csv
import os

import cudaq
import numpy as np

import nvtx

# import the circuits from circuits.py
from circuits import QAOA, QuantumVolume, QFT, CounterfeitCoin, GHZ

CIRCUIT_CLASSES = [GHZ, QAOA, QuantumVolume, QFT, CounterfeitCoin]

def eprint(*args, **kwargs):
    """
    Print to stderr.
    """
    print(*args, **kwargs, file=sys.stderr)


def print_result(result, num_qubits):
    """
    Print a histogram of the result
    """
    if num_qubits > 10:
        eprint('WARNING: cannot print results, num_qubits > 10')
        return

    d = {i: 0 for i in range(2**num_qubits)}
    d.update({int(k, 2): v for k, v in result.items()})

    print('\n'.join(f'{k} {v}' for k, v in d.items()))

def summarize_times(ts):
    return {
        'mean': float(np.mean(ts)),
        'median': float(np.median(ts)),
        'min': float(np.min(ts)),
        'max': float(np.max(ts)),
        'stddev': float(np.std(ts)),
    }

def write_csv_row(path, row):
    fieldnames = [
        'implementation', 'circuit', 'backend', 'qubits', 'shots',
        'repetitions', 'warmups', 'seed',
        'sample_mean_seconds', 'sample_median_seconds', 'sample_min_seconds',
        'sample_max_seconds', 'sample_stddev_seconds',
        'total_mean_seconds', 'total_median_seconds', 'total_min_seconds',
        'total_max_seconds', 'total_stddev_seconds',
        'program_total_seconds',
    ]

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def parse_qft_input_state(value):
    if value is None:
        return None
    if any(c not in '01' for c in value):
        raise ValueError('--qft-input-state must be a bit string, e.g. 01011')
    return [int(c) for c in value]

def parse_qaoa_weights(value):
    if value is None:
        return None
    try:
        weights = [int(item) for item in value.split(',') if item != '']
    except ValueError as exc:
        raise ValueError('--qaoa-weights must be a comma-separated list of -1/1') from exc
    if any(weight not in (-1, 1) for weight in weights):
        raise ValueError('--qaoa-weights may only contain -1 and 1')
    return weights

def run_one_experiment(num_qubits, num_shots, circuit_class, print_histo=False):
    program_start = time.perf_counter()
  
    if circuit_class is QFT and args.qft_input_state is not None:
        if len(args.qft_input_state) != num_qubits:
            raise ValueError('--qft-input-state length must equal --num-qubits')
        circuit = circuit_class(num_qubits, input_state=args.qft_input_state)
    elif circuit_class is QAOA and args.qaoa_weights is not None:
        expected = num_qubits * (num_qubits - 1) // 2
        if len(args.qaoa_weights) != expected:
            raise ValueError(f'--qaoa-weights length must be {expected}')
        circuit = circuit_class(num_qubits, rs=args.qaoa_weights)
    else:
        circuit = circuit_class(num_qubits)
    # --- instantiate circuits ---
    # circuit = QuantumVolume(num_qubits)
    # circuit = QAOA(num_qubits, np.pi/3, np.pi/6)
    # circuit = CounterfeitCoin(num_qubits)
    # circuit = QFT(num_qubits)
    #circuit = GHZ(num_qubits)

    # --- get kernel ---
    k, params = circuit.kernel, circuit.kernel_params

    sample_ts = []
    total_ts = []
    # --- main experiment loop ---
    for i in range(args.warmup + args.iter):
        r = nvtx.start_range(f'it{i}')
        total_t0 = time.perf_counter()
        sample_t0 = time.perf_counter()
        result = cudaq.sample(k, *params, 
                              shots_count=num_shots)
        sample_t1 = time.perf_counter()

        if print_histo:
            print(f'Most likely outcome: {result.most_probable()}')
            print_result(result, num_qubits)

        total_t1 = time.perf_counter()
        nvtx.end_range(r)
        
        sample_t = sample_t1 - sample_t0
        total_t = total_t1 - total_t0
        if i >= args.warmup:
            sample_ts.append(sample_t)
            total_ts.append(total_t)
        eprint(sample_t) # for debug

    sample_stats = summarize_times(sample_ts)
    total_stats = summarize_times(total_ts)
    avg = sample_stats['mean']
    std = sample_stats['stddev']
    print(f'{num_qubits} {avg} {std}')

    if args.csv_path:
        write_csv_row(args.csv_path, {
            'implementation': 'python',
            'circuit': circuit_class.__name__,
            'backend': args.target or '',
            'qubits': num_qubits,
            'shots': num_shots,
            'repetitions': args.iter,
            'warmups': args.warmup,
            'seed': args.seed if args.seed is not None else '',
            'sample_mean_seconds': sample_stats['mean'],
            'sample_median_seconds': sample_stats['median'],
            'sample_min_seconds': sample_stats['min'],
            'sample_max_seconds': sample_stats['max'],
            'sample_stddev_seconds': sample_stats['stddev'],
            'total_mean_seconds': total_stats['mean'],
            'total_median_seconds': total_stats['median'],
            'total_min_seconds': total_stats['min'],
            'total_max_seconds': total_stats['max'],
            'total_stddev_seconds': total_stats['stddev'],
            'program_total_seconds': time.perf_counter() - program_start,
        })

if __name__ == '__main__':
    # we need to set it manually, since CUDA-Q disables it by default
    sys.tracebacklimit = 1000

    p = argparse.ArgumentParser()
    p.add_argument('-n', '--num-qubits',   
                   type=int, required=True)
    p.add_argument('-N', '--num-qubits-max',
                   type=int, required=False,
                   help='if set, launches several repeat with `num_qubits` from `num_qubits` to `num_qubits_max`')
    p.add_argument('-s', '--num-shots',   
                   type=int, default=1024)
    p.add_argument('-w', '--warmup',
                   type=int, default=1,
                   help='numer of warmup iterations (default: 1)')
    p.add_argument('-i', '--iter',
                   type=int, default=10,
                   help='number of iterations (default: 10)')
    p.add_argument('--repetitions',
                   type=int, dest='iter',
                   help=argparse.SUPPRESS)
    # NOTE: `target` is parsed automatically by CUDA-Q, we do not need to handle it.
    p.add_argument('--target',
                   type=str, required=False,
                   help='target for CUDA-Q execution')

    circuit_names = [str(c.__name__) for c in CIRCUIT_CLASSES]
    p.add_argument('--circuit',
                   type=str, default=circuit_names[0], choices=circuit_names,
                   help=f'Circuit name to benchmark ({circuit_names}), default: {circuit_names[0]}.')
    p.add_argument('--seed',
                   type=int, required=False,
                   help='seed for random number generation (both NumPy and CUDA-Q)')
    p.add_argument('--histo',
                   action='store_true',
                   help='outputs a histogram of measurements for all quantum states')
    p.add_argument('--csv', '--output-csv',
                   dest='csv_path',
                   help='optional CSV path for summary timing results')
    p.add_argument('--qft-input-state',
                   type=parse_qft_input_state,
                   help='optional QFT input bit string used for Python/C++ parity')
    p.add_argument('--qaoa-weights',
                   type=parse_qaoa_weights,
                   help='optional comma-separated QAOA weights, each -1 or 1')

    args = p.parse_args()
    num_qubits_min = args.num_qubits
    num_qubits_max = (args.num_qubits_max or num_qubits_min) + 1
    num_shots  = args.num_shots
    circuit_class = CIRCUIT_CLASSES[circuit_names.index(args.circuit)]

    if args.seed is not None:
        np.random.seed(args.seed)
        cudaq.set_random_seed(args.seed)

    if args.target is not None:
        cudaq.set_target(args.target)

    eprint(args)

    for num_qubits in range(num_qubits_min, num_qubits_max):
        run_one_experiment(num_qubits, num_shots, circuit_class, args.histo)
    
