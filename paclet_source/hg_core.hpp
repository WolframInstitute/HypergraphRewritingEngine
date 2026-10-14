#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <functional>

// The bridge through which the host-agnostic evolution core reports progress.
// The callback may be empty. There is no cooperative abort: the standalone
// binary is aborted by the parent killing the process.
//
//   - Standalone binary: progress -> stderr.
//   - LibraryLink fallback: progress empty (no in-process progress).
struct HostBridge {
    std::function<void(const std::string&)> progress;
};

// A GPU engine binary that cannot use its device refuses every job with a message that starts
// with this prefix and continues with the reason (hg_gpu::DeviceProbe::reason). A worker sends
// the message as an error frame; the one-shot mode writes it to stderr and exits with
// kNoUsableDeviceExit. The paclet kernel matches the same prefix (hgNoDevicePrefix in
// HypergraphRewriting.wl) and runs the job on the CPU.
inline constexpr char kNoUsableDevicePrefix[] = "no usable GPU: ";
inline constexpr int kNoUsableDeviceExit = 3;

// Parse a WXF input association, run the multiway evolution, and serialize the
// result association to WXF. Input and output byte streams are identical to the
// LibraryLink performRewriting contract. Throws std::exception on error.
std::vector<uint8_t> run_rewriting_core(const std::vector<uint8_t>& wxf_bytes,
                                        const HostBridge& host);
