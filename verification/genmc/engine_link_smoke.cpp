// GENMC-LINK: engine
//
// Smoke test for the COMPOSED-ENGINE build path, not a property of the engine.
//
// It links every engine translation unit and touches one structure, so it fails if the linking
// path breaks -- a missing pthread declaration, an unresolved definition, an intrinsic the
// checker's code generator lacks, or a prune that removes main. Those are the four ways this
// pipeline has broken so far, and each was silent until something downstream timed out.
//
// The property is trivial on purpose: a fresh SegmentedArray is empty. What is under test is that
// GenMC can be handed the whole engine and still finish.
#include "hypergraph/types.hpp"
#include "hypergraph/segmented_array.hpp"

#include <cassert>

int main() {
    hg::engine::SegmentedArray<hg::engine::Edge> s;
    assert(s.size() == 0);
    return 0;
}
