#include "hg_gpu/cuda_check.hpp"
#include "hg_gpu/errors.hpp"
#include "hg_gpu/overflow.hpp"
#include "hg_gpu/termination.hpp"

// The host bodies behind the port's error reporting: cuda_check.hpp's throw, overflow.hpp's
// kind names, and DeviceErrors' allocation and drain. None of it is device code and none of it
// is on a hot path -- it runs once per failing CUDA call or once per kernel sync -- but every
// one of the sixteen translation units that includes engine_state.hpp was compiling all of it.
//
// cuda_check_at itself is NOT here. It stays inline in cuda_check.hpp, where its one comparison
// is what every CUDA call in the port pays; this file holds the throw it jumps to.

#include <stdexcept>
#include <string>

namespace HG_NAMESPACE {
namespace gpu {

void cuda_fail(cudaError_t err, const char* what, const char* file, int line) {
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + " " + what + ": " +
                             cudaGetErrorString(err));
}

const char* error_kind_name(ErrorKind k) {
    switch (k) {
        case ErrorKind::kEdgePoolFull:        return "edge_pool";
        case ErrorKind::kVertexPoolFull:      return "vertex_pool";
        case ErrorKind::kEventPoolFull:       return "event_pool";
        case ErrorKind::kStatePoolFull:       return "state_pool (max_states)";
        case ErrorKind::kCausalPoolFull:      return "causal_edge_pool";
        case ErrorKind::kBranchialPoolFull:   return "branchial_edge_pool";
        case ErrorKind::kMatchPoolFull:       return "match_pool";
        case ErrorKind::kCausalTripleMapFull: return "causal_triple_dedup";
        case ErrorKind::kCausalPairMapFull:   return "causal_pair_dedup";
        case ErrorKind::kBranchialMapFull:    return "branchial_pair_dedup";
        case ErrorKind::kEdgeConsumerNodes:   return "edge_consumers (node pool)";
        case ErrorKind::kBranchialIndexNodes: return "branchial_index (node pool)";
        case ErrorKind::kTrPredsNodes:        return "tr_preds (node pool)";
        case ErrorKind::kQcNodes:             return "quotient-causal records/nodes";
        case ErrorKind::kSigIndexNodes:       return "signature_index (node pool)";
        case ErrorKind::kInvIndexNodes:       return "vertex_inverted_index (node pool)";
        case ErrorKind::kFrontierCapFull:     return "frontier buffer";
        case ErrorKind::kScratchOverflow:     return "per-thread scratch (TR/WL)";
        case ErrorKind::kIRArenaExhausted:    return "device IR arena (retryable: grow config)";
        case ErrorKind::kIRDepthExceeded:     return "IR search depth (retryable: grow config)";
        case ErrorKind::kIRGeneratorsExceeded: return "IR automorphism generators (retryable: grow config)";
        case ErrorKind::kDeviceOutOfMemory:   return "device memory (engine allocation)";
        case ErrorKind::kPersistentStall:     return "persistent scheduler spin budget (defect)";
        case ErrorKind::kCanonicalMapFull:    return "canonical dedup map (retryable: grow config)";
        case ErrorKind::kEventSigRawFallback:
            return "event signatures built from a raw edge id (not an isomorphism invariant)";
        default:                              return "unknown";
    }
}

// =============================================================================
// DeviceErrors
// =============================================================================

DeviceErrors::DeviceErrors() {
    cudaError_t err = cudaMalloc(&counters_, sizeof(uint32_t) * kMaxKinds);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("DeviceErrors alloc: ") +
                                 cudaGetErrorString(err));
    }
    clear();
}

DeviceErrors::~DeviceErrors() {
    if (counters_) cudaFree(counters_);
}

DeviceErrors::DeviceView DeviceErrors::view() const { return DeviceView{counters_}; }

void DeviceErrors::clear() {
    cudaMemset(counters_, 0, sizeof(uint32_t) * kMaxKinds);
}

DeviceErrors::PoolOverflow::PoolOverflow(ErrorKind k, uint32_t cnt, const std::string& full_msg)
    : std::runtime_error(full_msg), kind(k), count(cnt) {}

void DeviceErrors::collect_warnings_into(std::vector<OverflowWarning>& out,
                                         const char* context) {
    uint32_t host[kMaxKinds] = {};
    cudaError_t err = cudaMemcpy(host, counters_,
                                 sizeof(uint32_t) * kMaxKinds,
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("DeviceErrors d2h: ") +
                                 cudaGetErrorString(err));
    }
    bool any = false;
    for (uint32_t i = 0; i < kMaxKinds; ++i) {
        if (host[i] == 0) continue;
        out.push_back(OverflowWarning{
            static_cast<ErrorKind>(i),
            host[i],
            std::string(context),
        });
        any = true;
    }
    if (any) {
        cudaMemset(counters_, 0, sizeof(uint32_t) * kMaxKinds);
    }
}

void DeviceErrors::throw_if_any(const char* context) const {
    uint32_t host[kMaxKinds] = {};
    cudaError_t err = cudaMemcpy(host, counters_,
                                 sizeof(uint32_t) * kMaxKinds,
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("DeviceErrors d2h: ") +
                                 cudaGetErrorString(err));
    }
    std::string msg;
    ErrorKind first_kind = ErrorKind::kCount;  // sentinel
    uint32_t  first_count = 0;
    for (uint32_t i = 0; i < kMaxKinds; ++i) {
        if (host[i] == 0) continue;
        if (first_kind == ErrorKind::kCount) {
            first_kind = static_cast<ErrorKind>(i);
            first_count = host[i];
        }
        if (!msg.empty()) msg += "; ";
        msg += error_kind_name(static_cast<ErrorKind>(i));
        msg += " overflowed ";
        msg += std::to_string(host[i]);
        msg += " times";
    }
    if (first_kind != ErrorKind::kCount) {
        throw PoolOverflow(first_kind, first_count,
            std::string("hg_gpu capacity overflow during ") + context +
            ": " + msg + ". Raise the corresponding EngineConfig field.");
    }
}


// =============================================================================
// TerminationDetector -- host side
// =============================================================================
//
// Here with the error reporting because it is the same kind of code: host-side allocation and
// counter drain around a device view, run once per launch rather than per item. The device
// view's mark_pushed / mark_completed / snapshot_quiescent stay in the header, being __device__.

TerminationDetector::TerminationDetector(uint32_t num_roles) : num_roles_(num_roles) {
    if (num_roles_ == 0 || num_roles_ > kMaxRoles) {
        throw std::invalid_argument("TerminationDetector num_roles out of range");
    }
    HG_CUDA_CHECK(cudaMalloc(&pushed_,      sizeof(uint64_t) * kMaxRoles), "TD pushed alloc");
    HG_CUDA_CHECK(cudaMalloc(&completed_,   sizeof(uint64_t) * kMaxRoles), "TD completed alloc");
    HG_CUDA_CHECK(cudaMalloc(&should_exit_, sizeof(uint32_t)),             "TD should_exit alloc");
    clear();
}

TerminationDetector::~TerminationDetector() {
    if (pushed_)      cudaFree(pushed_);
    if (completed_)   cudaFree(completed_);
    if (should_exit_) cudaFree(should_exit_);
}

TerminationDetector::DeviceView TerminationDetector::view() {
    return DeviceView{pushed_, completed_, should_exit_, num_roles_};
}

void TerminationDetector::clear() {
    HG_CUDA_CHECK(cudaMemset(pushed_,      0, sizeof(uint64_t) * kMaxRoles), "TD clear pushed");
    HG_CUDA_CHECK(cudaMemset(completed_,   0, sizeof(uint64_t) * kMaxRoles), "TD clear completed");
    HG_CUDA_CHECK(cudaMemset(should_exit_, 0, sizeof(uint32_t)),             "TD clear should_exit");
}

void TerminationDetector::mark_pushed_host(uint32_t role, uint64_t n) {
    if (role >= num_roles_) throw std::invalid_argument("TD mark_pushed_host bad role");
    uint64_t v = 0;
    HG_CUDA_CHECK(cudaMemcpy(&v, pushed_ + role, sizeof(uint64_t), cudaMemcpyDeviceToHost),
          "TD read pushed");
    v += n;
    HG_CUDA_CHECK(cudaMemcpy(pushed_ + role, &v, sizeof(uint64_t), cudaMemcpyHostToDevice),
          "TD write pushed");
}

bool TerminationDetector::exit_requested_host() const {
    uint32_t v = 0;
    cudaMemcpy(&v, should_exit_, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return v != 0;
}

uint32_t TerminationDetector::num_roles() const { return num_roles_; }

}  // namespace gpu
}  // namespace HG_NAMESPACE
