#include <events/viz_event_sink.hpp>

namespace viz {

// Static member definitions - must be in a single translation unit
// to avoid ODR violations when linking hypergraph library with visualization
MPSCRingBuffer<VizEvent>* VizEventSink::buffer_ = nullptr;
std::atomic<uint64_t> VizEventSink::events_emitted_{0};
std::atomic<uint64_t> VizEventSink::events_dropped_{0};

} // namespace viz
