// A thread_local object with a destructor registers it through __cxa_thread_atexit. The checker
// treats that call as an internal function that records nothing, so the thread runs to its end
// and the assertion after the join is checked.
#include <cassert>
#include <pthread.h>

struct D {
    int v = 1;
    ~D() {}
};
thread_local D d;
int done = 0;

void* t(void*) {
    done = d.v;
    return nullptr;
}

int main() {
    pthread_t th;
    pthread_create(&th, nullptr, t, nullptr);
    pthread_join(th, nullptr);
    assert(done == 1);
    return 0;
}
