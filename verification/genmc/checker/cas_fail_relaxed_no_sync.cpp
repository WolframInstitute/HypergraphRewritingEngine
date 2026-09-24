// Expect: Non-atomic race
// The CAS has success ordering seq_cst and failure ordering relaxed, and expects 5, so reading
// pub's 1 fails. The failed read is relaxed: it does not synchronise with the release store,
// and the read of `data` races with pub's write.
#include <pthread.h>
#include <atomic>
int data;
std::atomic<int> flag{0};
void* cas(void*) {
  int exp = 5;
  if (!flag.compare_exchange_strong(exp, 7, std::memory_order_seq_cst, std::memory_order_relaxed) && exp == 1)
    return (void*)(long)data;
  return nullptr; }
void* pub(void*) { data = 1; flag.store(1, std::memory_order_release); return nullptr; }
int main() { pthread_t t1, t2; pthread_create(&t1, 0, cas, 0); pthread_create(&t2, 0, pub, 0);
  pthread_join(t1, 0); pthread_join(t2, 0); return 0; }
