// A loop whose flag is true on entry and whose first body block merges two paths
// (tests/correct/saver/loop-body-phi upstream). After LoopJumpThreadingPass sends the entry into
// that block, the PHI of v needs an entry for the new edge from the preheader. Without the fix
// the PHI has none: the checker stops on a segmentation fault after the transform, and
// `opt -passes=verify` on the module reports "PHINode should have one entry for each predecessor
// of its parent basic block!".
#include <cassert>
#include <pthread.h>

int n, sum;

static int g() { return 2; }
static bool done() { return ++n == 3; }

void* thread_1(void*) {
    bool first = true;
    for (;;) {
        int v = 1;
        if (!first) v = g();
        sum += v;
        first = false;
        if (done()) break;
    }
    return nullptr;
}

int main() {
    pthread_t t;
    pthread_create(&t, nullptr, thread_1, nullptr);
    pthread_join(t, nullptr);
    assert(sum == 5);
    return 0;
}
