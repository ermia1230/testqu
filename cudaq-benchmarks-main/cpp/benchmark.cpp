#include "kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string circuit = "GHZ";
  int qubit_count = 0;
  std::size_t shots = 1024;
  int repetitions = 10;
  int warmups = 1;
  std::string backend = "compile-time";
  std::string csv_path;
  std::uint64_t seed = 0;
  bool has_seed = false;
  std::optional<std::vector<int>> qft_input_state;
  std::optional<std::vector<int>> qaoa_weights;
};

struct TimedRun {
  double sample_seconds = 0.0;
  double total_seconds = 0.0;
  std::size_t result_size = 0;
};

struct Stats {
  double mean = 0.0;
  double median = 0.0;
  double min = 0.0;
  double max = 0.0;
  double stddev = 0.0;
};

void print_help(const char *program) {
  std::cout
      << "Usage: " << program << " --circuit GHZ -n 20 -s 1024 -i 10 [options]\n\n"
      << "Options:\n"
      << "  --circuit NAME          Circuit: GHZ, QFT, QAOA (default: GHZ)\n"
      << "  -n, --num-qubits N      Number of qubits (required)\n"
      << "  -s, --num-shots N       Number of shots (default: 1024)\n"
      << "      --shots N           Alias for --num-shots\n"
      << "  -i, --iter N            Timed repetitions (default: 10)\n"
      << "      --repetitions N     Alias for --iter\n"
      << "  -w, --warmup N          Warmup repetitions (default: 1)\n"
      << "      --target NAME       Backend label for output. The CUDA-Q C++ target\n"
      << "                          is selected when compiling with nvq++ --target.\n"
      << "      --backend NAME      Alias for --target\n"
      << "      --csv PATH          Append summary CSV row to PATH\n"
      << "      --output-csv PATH   Alias for --csv\n"
      << "      --seed N            Seed C++ host-side parameter generation\n"
      << "      --qft-input-state BITS\n"
      << "                          Optional QFT input bit string for parity runs\n"
      << "      --qaoa-weights CSV  Optional comma-separated -1/1 QAOA weights\n"
      << "  -h, --help              Show this help\n";
}

int parse_int(const std::string &name, const std::string &value) {
  std::size_t consumed = 0;
  const int parsed = std::stoi(value, &consumed);
  if (consumed != value.size())
    throw std::invalid_argument("invalid integer for " + name + ": " + value);
  return parsed;
}

std::size_t parse_size(const std::string &name, const std::string &value) {
  std::size_t consumed = 0;
  const auto parsed = std::stoull(value, &consumed);
  if (consumed != value.size())
    throw std::invalid_argument("invalid size for " + name + ": " + value);
  return static_cast<std::size_t>(parsed);
}

std::uint64_t parse_u64(const std::string &name, const std::string &value) {
  std::size_t consumed = 0;
  const auto parsed = std::stoull(value, &consumed);
  if (consumed != value.size())
    throw std::invalid_argument("invalid integer for " + name + ": " + value);
  return static_cast<std::uint64_t>(parsed);
}

std::vector<int> parse_qft_input_state(const std::string &value) {
  std::vector<int> bits;
  bits.reserve(value.size());
  for (const auto c : value) {
    if (c != '0' && c != '1')
      throw std::invalid_argument("--qft-input-state must be a bit string");
    bits.push_back(c == '1' ? 1 : 0);
  }
  return bits;
}

std::vector<int> parse_qaoa_weights(const std::string &value) {
  std::vector<int> weights;
  std::stringstream stream(value);
  std::string item;

  while (std::getline(stream, item, ',')) {
    if (item.empty())
      continue;
    const int weight = parse_int("--qaoa-weights", item);
    if (weight != -1 && weight != 1)
      throw std::invalid_argument("--qaoa-weights may only contain -1 and 1");
    weights.push_back(weight);
  }

  return weights;
}

Options parse_args(int argc, char **argv) {
  Options options;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc)
        throw std::invalid_argument(name + " requires a value");
      return argv[++i];
    };

    if (arg == "-h" || arg == "--help") {
      print_help(argv[0]);
      std::exit(0);
    } else if (arg == "--circuit") {
      options.circuit = cudaq_bench::canonical_circuit_name(need_value(arg));
    } else if (arg == "-n" || arg == "--num-qubits") {
      options.qubit_count = parse_int(arg, need_value(arg));
    } else if (arg == "-s" || arg == "--num-shots" || arg == "--shots") {
      options.shots = parse_size(arg, need_value(arg));
    } else if (arg == "-i" || arg == "--iter" || arg == "--repetitions") {
      options.repetitions = parse_int(arg, need_value(arg));
    } else if (arg == "-w" || arg == "--warmup") {
      options.warmups = parse_int(arg, need_value(arg));
    } else if (arg == "--target" || arg == "--backend") {
      options.backend = need_value(arg);
    } else if (arg == "--csv" || arg == "--output-csv") {
      options.csv_path = need_value(arg);
    } else if (arg == "--seed") {
      options.seed = parse_u64(arg, need_value(arg));
      options.has_seed = true;
    } else if (arg == "--qft-input-state") {
      options.qft_input_state = parse_qft_input_state(need_value(arg));
    } else if (arg == "--qaoa-weights") {
      options.qaoa_weights = parse_qaoa_weights(need_value(arg));
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }

  if (options.qubit_count <= 0)
    throw std::invalid_argument("--num-qubits must be a positive integer");
  if (options.repetitions <= 0)
    throw std::invalid_argument("--iter/--repetitions must be positive");
  if (options.warmups < 0)
    throw std::invalid_argument("--warmup must be non-negative");
  if (options.shots == 0)
    throw std::invalid_argument("--num-shots must be positive");
  if (options.qft_input_state &&
      static_cast<int>(options.qft_input_state->size()) != options.qubit_count)
    throw std::invalid_argument(
        "--qft-input-state length must equal --num-qubits");
  if (options.qaoa_weights) {
    const int expected = options.qubit_count * (options.qubit_count - 1) / 2;
    if (static_cast<int>(options.qaoa_weights->size()) != expected)
      throw std::invalid_argument("--qaoa-weights length must equal n*(n-1)/2");
  }

  options.circuit = cudaq_bench::canonical_circuit_name(options.circuit);
  return options;
}

double elapsed_seconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

Stats summarize(std::vector<double> values) {
  if (values.empty())
    return {};

  Stats stats;
  stats.min = *std::min_element(values.begin(), values.end());
  stats.max = *std::max_element(values.begin(), values.end());
  stats.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();

  std::sort(values.begin(), values.end());
  const auto mid = values.size() / 2;
  stats.median = values.size() % 2 == 0 ? (values[mid - 1] + values[mid]) / 2.0
                                        : values[mid];

  double variance = 0.0;
  for (const auto value : values) {
    const double delta = value - stats.mean;
    variance += delta * delta;
  }
  stats.stddev = std::sqrt(variance / values.size());
  return stats;
}

TimedRun run_sample(const Options &options, const std::vector<int> &qft_input,
                    const std::vector<int> &qaoa_weights) {
  TimedRun timed;
  const auto total_start = Clock::now();
  const auto sample_start = Clock::now();

  if (options.circuit == "GHZ") {
    auto result = cudaq::sample(options.shots, cudaq_bench::GHZKernel{},
                                options.qubit_count);
    timed.result_size = result.size();
  } else if (options.circuit == "QFT") {
    auto result =
        cudaq::sample(options.shots, cudaq_bench::QFTKernel{}, qft_input);
    timed.result_size = result.size();
  } else if (options.circuit == "QAOA") {
    constexpr double gamma = 1.04719755119659774615;
    constexpr double beta = 0.52359877559829887308;
    auto result = cudaq::sample(options.shots, cudaq_bench::QAOAKernel{},
                                options.qubit_count, qaoa_weights, gamma, beta);
    timed.result_size = result.size();
  } else {
    throw std::invalid_argument("unsupported circuit: " + options.circuit);
  }

  const auto sample_end = Clock::now();
  const auto total_end = Clock::now();
  timed.sample_seconds = elapsed_seconds(sample_start, sample_end);
  timed.total_seconds = elapsed_seconds(total_start, total_end);
  return timed;
}

void write_csv(const Options &options, const Stats &sample_stats,
               const Stats &total_stats, double program_total_seconds) {
  if (options.csv_path.empty())
    return;

  const std::filesystem::path csv_path(options.csv_path);
  if (csv_path.has_parent_path())
    std::filesystem::create_directories(csv_path.parent_path());

  const bool exists = static_cast<bool>(std::ifstream(options.csv_path));
  std::ofstream csv(options.csv_path, std::ios::app);
  if (!csv)
    throw std::runtime_error("failed to open CSV path: " + options.csv_path);

  if (!exists) {
    csv << "implementation,circuit,backend,qubits,shots,repetitions,warmups,seed,"
        << "sample_mean_seconds,sample_median_seconds,sample_min_seconds,"
        << "sample_max_seconds,sample_stddev_seconds,total_mean_seconds,"
        << "total_median_seconds,total_min_seconds,total_max_seconds,"
        << "total_stddev_seconds,program_total_seconds\n";
  }

  csv << std::setprecision(17) << "cpp," << options.circuit << ','
      << options.backend << ',' << options.qubit_count << ',' << options.shots
      << ',' << options.repetitions << ',' << options.warmups << ','
      << (options.has_seed ? std::to_string(options.seed) : "") << ','
      << sample_stats.mean << ',' << sample_stats.median << ','
      << sample_stats.min << ',' << sample_stats.max << ','
      << sample_stats.stddev << ',' << total_stats.mean << ','
      << total_stats.median << ',' << total_stats.min << ',' << total_stats.max
      << ',' << total_stats.stddev << ',' << program_total_seconds << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto program_start = Clock::now();
    const auto options = parse_args(argc, argv);

    const auto qft_input =
        options.qft_input_state
            ? *options.qft_input_state
            : cudaq_bench::make_qft_input_state(options.qubit_count,
                                                options.seed, options.has_seed);
    const auto qaoa_weights =
        options.qaoa_weights
            ? *options.qaoa_weights
            : cudaq_bench::make_qaoa_weights(options.qubit_count, options.seed,
                                             options.has_seed);

    std::vector<double> sample_times;
    std::vector<double> total_times;
    sample_times.reserve(options.repetitions);
    total_times.reserve(options.repetitions);

    std::size_t last_result_size = 0;
    for (int i = 0; i < options.warmups + options.repetitions; ++i) {
      const auto timed = run_sample(options, qft_input, qaoa_weights);
      last_result_size = timed.result_size;

      if (i >= options.warmups) {
        sample_times.push_back(timed.sample_seconds);
        total_times.push_back(timed.total_seconds);
      }

      std::cerr << timed.sample_seconds << '\n';
    }

    const auto sample_stats = summarize(sample_times);
    const auto total_stats = summarize(total_times);
    const auto program_total_seconds =
        elapsed_seconds(program_start, Clock::now());

    std::cout << std::setprecision(10) << options.qubit_count << ' '
              << sample_stats.mean << ' ' << sample_stats.stddev << '\n';
    std::cerr << "result_size=" << last_result_size << '\n';

    write_csv(options, sample_stats, total_stats, program_total_seconds);
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
