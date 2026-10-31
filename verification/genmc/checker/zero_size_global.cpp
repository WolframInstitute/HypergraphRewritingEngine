// A global of zero size: an external array of unknown bound whose address is taken, as
// libstdc++'s std::function takes the address of __cxxabiv1::__class_type_info's vtable (an
// external [0 x ptr]) for target_type(). The checker stopped on "Internal check failed:
// size != 0" in SAddrAllocator::allocate while giving it a static address.
#include <cassert>

extern "C" char g_zero[];
char g_zero[] = {};

int main() {
    const char* p = g_zero;
    assert(p != nullptr);
    return 0;
}
