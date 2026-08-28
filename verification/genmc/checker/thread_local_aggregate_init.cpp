#include <pthread.h>
#include <atomic>
/* A function-scope thread_local with a destructor: the compiler emits a one-byte guard
 * object, a thread_local global of an aggregate type ({ i8 }) with a zeroinitializer. The
 * checker materialises a thread-local's initializer as one value per byte through
 * getConstantValue, which has no case for an aggregate: "Constant unimplemented for type". */
void* __dso_handle = nullptr;   /* the thread-local destructor registration names it */
std::atomic<int> x{0};
struct Guard { int v = 0; ~Guard() { x.fetch_add(v, std::memory_order_relaxed); } };
int* slot() { static thread_local Guard g; g.v = 1; static thread_local int cell = 0; return &cell; }
void* w(void*) { *slot() += 1; return nullptr; }
int main() { pthread_t t; pthread_create(&t, 0, w, 0); *slot() += 2; pthread_join(t, 0); return 0; }
