#include <cassert>
#include <pthread.h>
struct C { void* b; unsigned long o, c, n; };
C* cs;
void* t(void*) { cs[1].o = 7; return nullptr; }
int main() { cs = new C[4]; cs[0].o = 1; pthread_t th; pthread_create(&th, 0, t, 0); pthread_join(th, 0); assert(cs[1].o == 7 && cs[0].o == 1); delete[] cs; return 0; }
