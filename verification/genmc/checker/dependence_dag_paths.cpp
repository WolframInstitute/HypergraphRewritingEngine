#include <pthread.h>
#include <atomic>
/* A loop whose header condition depends on a value with an exponential number of operand paths.
 * b_n = a_(n-1) + b_(n-1), a_n = b_(n-1): the dependence DAG of b_40 has Fibonacci(40) paths
 * back to the load. LoopJumpThreadingPass asks whether the header's branch depends on the
 * header PHI (i); the branch's first operand is b_40, so a search that enumerates paths walks
 * every one of them before it reaches i. The patched checker's reachability search visits each
 * of the 80 values once. */
std::atomic<int> x{0};
#define STEP(p, n) int a##n = b##p; int b##n = a##p + b##p;
void* w(void*) { x.store(1, std::memory_order_release); return nullptr; }
int main() {
  pthread_t t; pthread_create(&t, 0, w, 0);
  int a0 = x.load(std::memory_order_acquire), b0 = 1;
  STEP(0,1)
  STEP(1,2)
  STEP(2,3)
  STEP(3,4)
  STEP(4,5)
  STEP(5,6)
  STEP(6,7)
  STEP(7,8)
  STEP(8,9)
  STEP(9,10)
  STEP(10,11)
  STEP(11,12)
  STEP(12,13)
  STEP(13,14)
  STEP(14,15)
  STEP(15,16)
  STEP(16,17)
  STEP(17,18)
  STEP(18,19)
  STEP(19,20)
  STEP(20,21)
  STEP(21,22)
  STEP(22,23)
  STEP(23,24)
  STEP(24,25)
  STEP(25,26)
  STEP(26,27)
  STEP(27,28)
  STEP(28,29)
  STEP(29,30)
  STEP(30,31)
  STEP(31,32)
  STEP(32,33)
  STEP(33,34)
  STEP(34,35)
  STEP(35,36)
  STEP(36,37)
  STEP(37,38)
  STEP(38,39)
  STEP(39,40)
  int i = 0;
  while (b40 > i) { i += b40; }
  pthread_join(t, 0);
  return 0;
}
