#pragma once

// Per-device kernel-launch parameter cache, persisted as JSON next to the
// binary at hg_gpu_tuning_cache.json. Keyed by (compute_cap, sm_count,
// total_mem_GB, driver_version).
// Implementation in M9.1 / M9.6.
