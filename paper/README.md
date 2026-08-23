# Rewriting the Universe — paper

LaTeX source for "Rewriting the Universe: High-Performance Hypergraph Evolution for the Wolfram
Physics Project".

## Building

```bash
make        # full build with bibliography
make quick  # single pass, no bibliography
make clean  # remove auxiliary files
make arxiv  # arXiv submission tarball
```

## Layout

- `main.tex` — the paper
- `references.bib` — bibliography
- `tables/` — every table, figure fragment and numeric macro the paper includes. **All of them
  are generated**: each file's first line names the generator and its second line records the
  commit, the machine the numbers were measured on, the load at the start of the run and the
  core set the workers were pinned to. Edit the generator, not the file.

## Regenerating the numbers

```bash
python3 -u tools/dev/paper_tables.py --gpu --wolfram --authority-depth 7 --steps 7 \
        --cpus <homogeneous core set> --thread-sweep <counts within it>
python3 -u tools/dev/scaling_sweep.py --sections cpu,shapes,memory,gpu
cd paper && touch main.tex && make
```

`--cpus` pins the engine's workers to the named logical CPUs so a speedup column divides by a
homogeneous quantity of compute; on a hybrid CPU an unpinned thread count is not one. A table
generated on a machine carrying other load is stamped `*** CONTENDED` and is not for publication.
Two passages are prose about the data and are re-read by hand after regeneration: the C/R caption
and every sentence quoting T2's ratios.

## Sections

1. Introduction
2. Background and Related Work
3. Graph Canonicalization
4. Pattern Matching
5. System Architecture
6. GPU Acceleration
7. Verification of the Concurrent Surface
8. Experimental Evaluation
9. Conclusion and Future Work

Appendices: Complexity Analysis Details; Data Structure Specifications.
