// A memset whose length is known only at run time is lowered to a bounded loop; the bytes it
// writes read back as written.
#include <cassert>
#include <cstdlib>
#include <cstring>

int main() {
    volatile unsigned n = 12;
    unsigned char* p = static_cast<unsigned char*>(std::malloc(16));
    std::memset(p, 0xAB, n);
    assert(p[0] == 0xAB && p[11] == 0xAB);
    std::free(p);
    return 0;
}
