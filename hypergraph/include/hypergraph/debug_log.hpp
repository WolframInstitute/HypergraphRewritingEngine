#ifndef HYPERGRAPH_DEBUG_LOG_HPP
#define HYPERGRAPH_DEBUG_LOG_HPP

#include <cstdio>
#include <thread>
#include <sstream>

// Debug logging macro - thread-safe printf-based logging with thread ID
#ifdef ENABLE_DEBUG_OUTPUT
    #define DEBUG_LOG(fmt, ...) do { \
        std::ostringstream oss; \
        oss << std::this_thread::get_id(); \
        printf("[DEBUG][T%s] " fmt "\n", oss.str().c_str(), ##__VA_ARGS__); \
    } while(0)
#else
    #define DEBUG_LOG(fmt, ...) ((void)0)
#endif

#endif // HYPERGRAPH_DEBUG_LOG_HPP