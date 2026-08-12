# What this draft is, and what it is waiting for

**Tracked here so it cannot be lost again.** This source lived only in `RECOVERED_paper/`, a
directory excluded per-clone through `.git/info/exclude`, after the 2026-07-03 reconciliation
stranded it on an orphaned line and it had to be recovered from the reflog. A paper that exists
on one machine and in no commit is one `git gc` from gone, which is exactly what nearly happened.

## What is DONE

The two corrections the revision plan (`../docs/PAPER_REVISION_PLAN.md`) called the biggest
problems are applied:

- **Uniqueness trees are gone.** The abstract and the contribution list present McKay-style
  individualization–refinement as the exact reference canonicalizer and Weisfeiler–Leman as the
  fast hot-path hash with IR fallback. The `O(V^7)` claim, the runtime "hash-strategy selector"
  and the Bloom-filter incremental-UT subsection are not in this draft.
- **No fabricated numbers.** Every performance figure the earlier draft asserted was a projection
  or a placeholder. They are removed rather than adjusted, and each table that will carry a
  measurement instead carries a `\todo` naming the harness that must produce it.

## What it is WAITING for, and this is the whole of it

22 `\todo` markers, and they are not writing tasks. They are **measurements**, and the draft says
so itself: *"No performance number in this draft should be read as final."* Three of them need
instruments that do not exist yet:

| table | needs |
|---|---|
| T2 wall-time vs the Wolfram reference | a wall-time sweep harness (not built) |
| T9 per-contribution ablation | CMake `#define` ablation builds, old paths compiled out (not built) |
| T7 GPU acceleration | the GPU WL hash kernel's known `O(V·E)` inefficiency resolved or disclosed |

The rest need a **quiet machine**. `docs/PAPER_RESULTS.md` records the working numbers and says
plainly what they are worth: an RTX 4090 under WSL2 on a noisy box, CV 10–40%, *"treat as rough /
order-of-magnitude until re-run on a quiet machine."* The validation status in that file is
exact; the timings are not, and this box is the noisy one.

So the honest state is: the paper's ARGUMENT is written and its CLAIMS are true, and its
EVIDENCE section is a scaffold waiting on runs that this machine cannot produce credibly. That
is a different thing from "the paper is not written", and it is why the remaining work is
benchmarking rather than authoring.

## What is NOT here

`figures/` is not tracked: the draft's only figure reference is a placeholder screenshot, and the
build products (`main.pdf`, `.aux`, `.bbl`, `.log`) are outputs. `make` in this directory
rebuilds them.
