#include "circuits.hpp"

#include <algorithm>
#include <cctype>
#include <random>
#include <stdexcept>

namespace cudaq_bench {

namespace {

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::mt19937 make_rng(std::uint64_t seed, bool has_seed) {
  if (has_seed)
    return std::mt19937(static_cast<std::mt19937::result_type>(seed));

  std::random_device rd;
  return std::mt19937(rd());
}

} // namespace

std::string canonical_circuit_name(const std::string &name) {
  const auto value = lower(name);
  if (value == "ghz")
    return "GHZ";
  if (value == "qft")
    return "QFT";
  if (value == "qaoa")
    return "QAOA";

  throw std::invalid_argument("unsupported circuit '" + name +
                              "'; supported circuits: GHZ, QFT, QAOA");
}

std::vector<std::string> supported_circuits() { return {"GHZ", "QFT", "QAOA"}; }

std::vector<int> make_qft_input_state(int qubit_count, std::uint64_t seed,
                                      bool has_seed) {
  auto rng = make_rng(seed, has_seed);
  std::uniform_int_distribution<int> bit(0, 1);

  std::vector<int> input_state(qubit_count);
  for (auto &value : input_state)
    value = bit(rng);

  return input_state;
}

std::vector<int> make_qaoa_weights(int qubit_count, std::uint64_t seed,
                                   bool has_seed) {
  auto rng = make_rng(seed, has_seed);
  std::uniform_int_distribution<int> pick(0, 1);

  const int weight_count = qubit_count * (qubit_count - 1) / 2;
  std::vector<int> weights(weight_count);
  for (auto &value : weights)
    value = pick(rng) == 0 ? -1 : 1;

  return weights;
}

} // namespace cudaq_bench
