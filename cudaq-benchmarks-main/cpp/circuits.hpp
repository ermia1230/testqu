#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cudaq_bench {

std::string canonical_circuit_name(const std::string &name);
std::vector<std::string> supported_circuits();
std::vector<int> make_qft_input_state(int qubit_count, std::uint64_t seed,
                                      bool has_seed);
std::vector<int> make_qaoa_weights(int qubit_count, std::uint64_t seed,
                                   bool has_seed);

} // namespace cudaq_bench
