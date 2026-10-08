// Expect: Attempt to access non-allocated memory
// A compare_exchange_weak retry loop whose failure path dereferences the value the CAS returned,
// with nobody else writing the word. Only a spurious failure enters that path, and it returns the
// expected value, null. The load after the CAS runs only on a failure, so the checker keeps the
// CAS weak and reports the null dereference.
#include <pthread.h>
#include <atomic>

struct Node { int v; };
std::atomic<Node*> head{nullptr};
Node mine{1};
int sink;

void* publish(void*) {
    Node* old = nullptr;
    while (!head.compare_exchange_weak(old, &mine, std::memory_order_release,
                                       std::memory_order_acquire))
        sink = old->v;
    return nullptr;
}

int main() {
    pthread_t t;
    pthread_create(&t, nullptr, publish, nullptr);
    pthread_join(t, nullptr);
    return 0;
}
