// A loop whose flag is true on entry, so LoopJumpThreadingPass sends the entry straight into the
// body, and whose flag is read after the loop (tests/correct/saver/loop-header-use upstream).
// With size 1 and cap 2 the loop exits from the body on its first iteration, before any run of
// the header. Without the fix the read of the flag after the loop has no definition on that path:
// the checker reports "Assertion violation: done && cap == 2", and `opt -passes=verify` on the
// module from --output-llvm-after reports "Instruction does not dominate all uses!".
#include <cassert>
#include <pthread.h>

unsigned size = 1;
unsigned cap = 2;
bool done;

static bool full() { return size > cap; }
static bool grow() { cap *= 2; return cap < 64; }

void* thread_1(void*) {
    bool ok = true;
    while (ok && full()) ok = grow();
    done = ok;
    return nullptr;
}

int main() {
    pthread_t t;
    pthread_create(&t, nullptr, thread_1, nullptr);
    pthread_join(t, nullptr);
    assert(done && cap == 2);
    return 0;
}
