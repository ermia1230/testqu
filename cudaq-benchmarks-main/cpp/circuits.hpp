#pragma once

#include <cudaq.h>

#include <cstdint>
#include <string>
#include <vector>

namespace cudaq_bench {

struct GHZKernel {
  void operator()(int qubit_count) __qpu__ {
    cudaq::qvector qvector(qubit_count);
    h(qvector[0]);

    for (int i = 0; i < qubit_count - 1; ++i)
      x<cudaq::ctrl>(qvector[i], qvector[i + 1]);

    mz(qvector);
  }
};

struct QFTKernel {
  void operator()(std::vector<int> input_state) __qpu__ {
    const int qubit_count = input_state.size();
    cudaq::qvector qubits(qubit_count);

    for (int i = 0; i < qubit_count; ++i) {
      if (input_state[i] == 1)
        x(qubits[i]);
    }

    for (int i = 0; i < qubit_count; ++i) {
      h(qubits[i]);
      for (int j = i + 1; j < qubit_count; ++j) {
        const double angle = 6.28318530717958647692 / (1ULL << (j - i + 1));
        r1<cudaq::ctrl>(angle, qubits[j], qubits[i]);
      }
    }

    // Match circuits.py: QFT has no explicit mz; cudaq::sample implicitly
    // samples all allocated qubits at the end of the kernel.
  }
};

struct QAOAKernel {
  void operator()(int qubit_count, std::vector<int> rs, double gamma,
                  double beta) __qpu__ {
    cudaq::qvector qvector(qubit_count);

    h(qvector);

    int c = 0;
    for (int i = 0; i < qubit_count; ++i) {
      for (int j = i + 1; j < qubit_count; ++j) {
        const double phi = gamma * rs[c];

        x<cudaq::ctrl>(qvector[i], qvector[j]);
        rz(2.0 * phi, qvector[j]);
        x<cudaq::ctrl>(qvector[i], qvector[j]);

        ++c;
      }
    }

    for (int i = 0; i < qubit_count; ++i)
      rx(2.0 * beta, qvector[i]);

    mz(qvector);
  }
};

std::string canonical_circuit_name(const std::string &name);
std::vector<std::string> supported_circuits();
std::vector<int> make_qft_input_state(int qubit_count, std::uint64_t seed,
                                      bool has_seed);
std::vector<int> make_qaoa_weights(int qubit_count, std::uint64_t seed,
                                   bool has_seed);

} // namespace cudaq_bench
