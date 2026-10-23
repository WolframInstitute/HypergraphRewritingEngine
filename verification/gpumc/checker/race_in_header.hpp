#pragma once
// Included by race_in_header.cpp: two plain stores in a file other than the harness.
#include <cstdint>

inline void race_in_header_store_one(uint32_t* p) { *p = 1; }
inline void race_in_header_store_two(uint32_t* p) { *p = 2; }
