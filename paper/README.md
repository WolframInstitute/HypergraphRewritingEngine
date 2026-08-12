# Rewriting the Universe - Paper

LaTeX source for the paper "Rewriting the Universe: High-Performance Hypergraph Evolution for the Wolfram Physics Project"

## Building

```bash
make        # Full build with bibliography
make quick  # Single pass (no bibliography)
make clean  # Remove auxiliary files
make arxiv  # Create arxiv submission tarball
```

## Structure

- `main.tex` - Main paper source
- `references.bib` - BibTeX bibliography
- `figures/` - Figures and plots (to be generated from benchmarks)

## Sections Overview

1. **Introduction** - Motivation and contributions
2. **Background** - Hypergraph rewriting, multiway evolution, related work
3. **Canonicalization** - Uniqueness trees, WL hashing, incremental computation
4. **Pattern Matching** - Signature indexing, task-based matching, match forwarding
5. **Architecture** - Unified storage, lock-free data structures, memory model
6. **GPU** - Megakernel design, parallel algorithms
7. **Benchmarks** - CPU/GPU performance, scaling, strategy comparison
8. **Visualization** - Interactive exploration tools
9. **Conclusion** - Summary and future work

## TODO

- [ ] Generate actual benchmark data and plots
- [ ] Add visualization screenshots
- [ ] Create architecture diagrams (TikZ)
- [ ] Fill in placeholder URLs
- [ ] Verify all citations
- [ ] Proofread and polish

## Figures Needed

1. CPU performance vs Wolfram Language (line plot)
2. GPU throughput by graph size (line plot)
3. Thread scaling efficiency (line plot)
4. Architecture diagram (storage, indices, data flow)
5. Megakernel architecture diagram
6. Uniqueness tree construction example
7. Pattern matching workflow
8. Interactive visualization screenshot(s)
