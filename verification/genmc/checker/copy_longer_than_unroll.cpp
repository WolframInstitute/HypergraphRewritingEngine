// Expect: longer than the --unroll bound
// Args: --unroll=8
// A memmove is lowered to a loop of one byte per iteration. At --unroll=8 a copy of 8 bytes
// reaches the bound, and the checker reports it at the memmove. Killing the thread there
// would end the execution with no error and leave the assertion below unchecked.
#include <cassert>
#include <cstring>
char src[8] = {1, 2, 3, 4, 5, 6, 7, 8};
char dst[8];
char *volatile dstp = dst;
int main() {
  memmove(dstp, src, 8);
  assert(dst[7] == 0);
  return 0; }
