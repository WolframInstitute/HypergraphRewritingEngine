// A global declared and not defined, resolved to the symbol of that name in the checker's own
// process, as libstdc++'s lambda typeinfo points into __cxxabiv1::__class_type_info's vtable for
// std::function::target_type(). The checker wrote the declaration's (absent) initializer over
// that symbol's storage and stopped on a segmentation fault before exploring.
#include <cassert>

extern void* class_type_info_vtable[] asm("_ZTVN10__cxxabiv117__class_type_infoE");
void* const* volatile g_p;

int main() {
    g_p = class_type_info_vtable + 2;
    assert(g_p != nullptr);
    return 0;
}
