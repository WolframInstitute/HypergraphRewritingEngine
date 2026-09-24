// A memmove of 6 bytes to 4 bytes above its source, over int-typed memory: lowered as one 4-byte
// word and a 2-byte tail, copied from the top since the destination lies inside the source. The
// destination pointer is loaded through a volatile, so the pass takes the word width from the
// source. The word's store covers the tail's first source byte, so the tail is copied first.
// The results are read at the widths and addresses the lowering writes them.
#include <cassert>
#include <cstring>
unsigned buf[3] = {0x04030201u, 0x08070605u, 0x0c0b0a09u};
char *volatile dstp = (char *)buf + 4;
int main() {
  char *d = dstp;
  memmove(d, buf, 6);
  assert(*(unsigned *)d == 0x04030201u);
  assert(d[4] == 5);
  assert(d[5] == 6);
  return 0; }
