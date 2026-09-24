// Expect: Non-atomic race
// The CAS has success ordering relaxed and failure ordering acquire. In the execution where it
// reads 1 from pub's release store it succeeds, its read is relaxed, and the read of `data`
// races with pub's write. The checker first explores the execution where it reads 0 and fails.
#include <pthread.h>
#include <atomic>
int data;
std::atomic<int> flag{0};
void* cas(void*) {
  int exp = 1;
  if (flag.compare_exchange_strong(exp, 2, std::memory_order_relaxed, std::memory_order_acquire))
    return (void*)(long)data;
  return nullptr; }
void* pub(void*) { data = 1; flag.store(1, std::memory_order_release); return nullptr; }
int main() { pthread_t t1, t2; pthread_create(&t1, 0, cas, 0); pthread_create(&t2, 0, pub, 0);
  pthread_join(t1, 0); pthread_join(t2, 0); return 0; }
