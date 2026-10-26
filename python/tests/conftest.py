import os
import sys

# The engine uses one worker per logical CPU unless told otherwise; the tests use four.
os.environ.setdefault("HG_TUNE_THREADS", "4")
# NumPy's OpenBLAS starts one thread per logical CPU; under an address-space cap (ulimit -v) on a
# many-core machine its buffers do not fit and the import stalls.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

# A build tree's package (python/hgevolve with the module beside it) when none is installed.
try:
    import hgevolve  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
