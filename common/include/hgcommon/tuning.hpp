#pragma once
#include "hgcommon/namespace.hpp"

// Expert overrides of the engine's derived tuning values (docs/TUNING.md, section 2.2).
//
// Each value has a default derived from the machine and the run. An override replaces it. The
// order is: the job's "Tuning" association (or "Threads"), then the HG_TUNE_<NAME> environment
// variable, then the derived default. tuned() applies that order, so every consumer resolves a
// value the same way. The environment is read once per process.
//
// Name, environment variable, unit, consumer:
//   Threads            HG_TUNE_THREADS               workers   ParallelEvolutionEngine, JobSystem
//   GPUMemoryLimit     HG_TUNE_GPU_MEMORY_LIMIT      bytes     resolved_memory_cap

#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace HG_NAMESPACE {
namespace common {

enum class Tune : uint8_t {
    Threads,
    GpuMemoryLimit,
    kCount
};

struct TuneInfo {
    const char* option;   // the key in "Tuning"
    const char* env;      // the environment variable
    int64_t min;          // the smallest value accepted
};

inline constexpr std::array<TuneInfo, static_cast<size_t>(Tune::kCount)> kTuneInfo{{
    {"Threads", "HG_TUNE_THREADS", 1},
    {"GPUMemoryLimit", "HG_TUNE_GPU_MEMORY_LIMIT", 1},
}};

inline const TuneInfo& tune_info(Tune t) { return kTuneInfo[static_cast<size_t>(t)]; }

class TuningOverrides {
public:
    std::optional<int64_t> get(Tune t) const {
        const size_t i = static_cast<size_t>(t);
        if (!has_[i]) return std::nullopt;
        return value_[i];
    }
    // Throws std::invalid_argument for a value below the name's minimum.
    void set(Tune t, int64_t v) {
        if (v < tune_info(t).min)
            throw std::invalid_argument(std::string(tune_info(t).option) + " must be at least " +
                                        std::to_string(tune_info(t).min) + ", got " +
                                        std::to_string(v));
        value_[static_cast<size_t>(t)] = v;
        has_[static_cast<size_t>(t)] = true;
    }
    bool empty() const {
        for (bool h : has_) if (h) return false;
        return true;
    }
    // The name a "Tuning" key spells, or nullopt.
    static std::optional<Tune> by_option(std::string_view name) {
        for (size_t i = 0; i < kTuneInfo.size(); ++i)
            if (name == kTuneInfo[i].option) return static_cast<Tune>(i);
        return std::nullopt;
    }
    // The names, comma-separated, for messages; `except` is left out.
    static std::string option_names(Tune except = Tune::kCount) {
        std::string s;
        for (size_t i = 0; i < kTuneInfo.size(); ++i)
            if (static_cast<Tune>(i) != except)
                s += (s.empty() ? "" : ", ") + std::string(kTuneInfo[i].option);
        return s;
    }

private:
    std::array<int64_t, static_cast<size_t>(Tune::kCount)> value_{};
    std::array<bool, static_cast<size_t>(Tune::kCount)> has_{};
};

// The HG_TUNE_* environment, read on the first call. A variable that does not parse as an
// integer at or above the name's minimum is ignored.
inline const TuningOverrides& tuning_env() {
    static const TuningOverrides env = [] {
        TuningOverrides o;
        for (size_t i = 0; i < kTuneInfo.size(); ++i) {
            const Tune t = static_cast<Tune>(i);
            const char* v = std::getenv(kTuneInfo[i].env);
            if (!v || !*v) continue;
            char* end = nullptr;
            const long long x = std::strtoll(v, &end, 10);
            if (end == v || *end != '\0' || x < kTuneInfo[i].min) continue;
            o.set(t, x);
        }
        return o;
    }();
    return env;
}

// Where a resolved value came from, for --print-tuning and the "Device" reply.
enum class TuneSource : uint8_t { Job, Environment, Derived };

inline const char* tune_source_name(TuneSource s) {
    switch (s) {
        case TuneSource::Job:         return "Tuning";
        case TuneSource::Environment: return "environment";
        case TuneSource::Derived:     return "derived";
    }
    return "derived";
}

// The value of `t`: the job's override, else the environment's, else derive().
template <class Derive>
int64_t tuned(const TuningOverrides& job, Tune t, Derive&& derive, TuneSource* source = nullptr) {
    if (auto v = job.get(t)) { if (source) *source = TuneSource::Job; return *v; }
    if (auto v = tuning_env().get(t)) { if (source) *source = TuneSource::Environment; return *v; }
    if (source) *source = TuneSource::Derived;
    return static_cast<int64_t>(derive());
}

}  // namespace common
}  // namespace HG_NAMESPACE
