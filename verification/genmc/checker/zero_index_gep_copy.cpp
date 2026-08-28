// A whole-array copy through a zero-index GEP. Before the promotion fix the checker took the
// GEP's result element type (i32) as the type to promote by and tripped typeSizeDst >= len;
// after it, the array type is used and the copy is two i32 stores, which is what the reads are.
#include <pthread.h>
#include <cassert>
#include <cstdint>
static uint32_t g[2];
void* t(void*) { uint32_t raws[2] = {10, 12}; g[0] = raws[0]; g[1] = raws[1]; return nullptr; }
int main() { pthread_t th; pthread_create(&th, 0, t, 0); pthread_join(th, 0); assert(g[0] == 10 && g[1] == 12); return 0; }
