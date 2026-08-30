#pragma once
#include "hgcommon/namespace.hpp"
#include <type_traits>

namespace HG_NAMESPACE {
namespace engine {
// T() WRITES ONLY ZERO BYTES. ConcurrentHeterogeneousArena::allocate_array skips the
// value-initialisation of such a T when the bytes it hands out are known to be zero already.
// True by default exactly for trivially default-constructible, trivially destructible types,
// whose T() is zero-initialisation by the language. A type whose default member initialisers
// are all zero says so by specialising this next to its definition; the arena test
// ArenaZeroInit.OptedInTypesAreZeroBytes checks every such type by constructing one.
template <typename T>
inline constexpr bool zero_value_init_v =
    std::is_trivially_default_constructible_v<T> && std::is_trivially_destructible_v<T>;

} // namespace engine
} // namespace HG_NAMESPACE
