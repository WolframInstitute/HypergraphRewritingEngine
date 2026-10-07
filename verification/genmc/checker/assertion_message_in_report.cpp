// Expect: the message this assertion carries
// An assertion's message is part of the error report, so a failure names which of several
// assertions in a harness fired.
#include <cassert>

int main() {
    volatile int x = 1;
    assert(x == 0 && "the message this assertion carries");
    return 0;
}
