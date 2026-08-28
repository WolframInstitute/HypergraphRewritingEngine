#include <pthread.h>
#include <atomic>
#include <cassert>
#include <cstdlib>
struct B { B* prev; B* next; long cap; };
std::atomic<B*> head{nullptr};
void* a(void*) { B* b = new B; b->prev = nullptr; b->next = nullptr; b->cap = 1;
  B* old = head.load(std::memory_order_acquire);
  while (!head.compare_exchange_weak(old, b, std::memory_order_release, std::memory_order_acquire)) {}
  b->prev = old; return nullptr; }
void* c(void*) { B* n = new B; n->prev = nullptr; n->next = nullptr; n->cap = 2;
  B* old = head.load(std::memory_order_acquire);
  while (!head.compare_exchange_weak(old, n, std::memory_order_release, std::memory_order_acquire)) {}
  n->prev = old; if (old) old->next = n;   /* write into the other thread's block, after acquire */
  return nullptr; }
int main() { pthread_t t1, t2; pthread_create(&t1, 0, a, 0); pthread_create(&t2, 0, c, 0);
  pthread_join(t1, 0); pthread_join(t2, 0); return 0; }
