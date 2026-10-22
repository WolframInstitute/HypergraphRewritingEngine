#pragma once
// Included ahead of every harness by run.sh. GPUMC compiles against its own C library headers.
//
// Its <pthread.h> declares threads, mutexes and barriers and no condition variables, keys or
// once-flags, which libstdc++'s POSIX thread layer (bits/gthr-posix.h, reached from <string>)
// names. libstdc++'s single-threaded layer is used instead: no harness uses a libstdc++ facility
// that synchronises.
//
// Its C headers declare no rand, srand, strto*, vsnprintf or errno, which libstdc++'s <algorithm>
// and <string> name. They are declared here. No harness calls them.
//
// run.sh puts libstdc++'s directories on the include path ahead of GPUMC's, whose C++ headers
// (<atomic>, <thread>) are an older libc++ and collide with libstdc++'s.
#include <bits/c++config.h>
#include <bits/gthr-single.h>
#define _GLIBCXX_GCC_GTHR_POSIX_H 1

#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>

extern "C" {
int rand(void);
void srand(unsigned);
long strtol(const char*, char**, int);
unsigned long strtoul(const char*, char**, int);
long long strtoll(const char*, char**, int);
unsigned long long strtoull(const char*, char**, int);
float strtof(const char*, char**);
double strtod(const char*, char**);
long double strtold(const char*, char**);
int vsnprintf(char*, size_t, const char*, va_list);
int* __errno_location(void);
}
#ifndef errno
#define errno (*__errno_location())
#endif

// EXCEPTIONS. run.sh compiles with -fno-exceptions: GPUMC cannot resolve a thrown type's
// typeinfo, which libstdc++ defines in its shared library. Device code throws nothing; the host
// functions of an included .cu do, and no harness calls them. Their `throw X;` evaluates X and
// discards it, `throw;` does nothing, a `try` block runs, and a `catch` block does not. The standard headers the engine
// includes are included first, so the three macros reach only the engine's code.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#define throw if (false); else
#define try if (true)
#define catch(...) if (false)

namespace std {
using ::rand;
using ::srand;
using ::strtol;
using ::strtoul;
using ::strtoll;
using ::strtoull;
using ::strtof;
using ::strtod;
using ::strtold;
using ::vsnprintf;
}  // namespace std
