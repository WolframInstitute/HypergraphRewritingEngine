import os
import sys

# The engine uses one worker per logical CPU unless told otherwise; the tests use four.
os.environ.setdefault("HG_TUNE_THREADS", "4")
# NumPy's OpenBLAS starts one thread per logical CPU; under an address-space cap (ulimit -v) on a
# many-core machine its buffers do not fit and the import stalls.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import pytest

# The package: as found on the path; else the installed one when python/hgevolve (the sources,
# without the compiled module) shadowed it, as it does when the tests run from python/; else the build
# tree's package (python/hgevolve with the module beside it).
SOURCE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    import hgevolve
except ImportError:
    for _m in [m for m in sys.modules if m == "hgevolve" or m.startswith("hgevolve.")]:
        del sys.modules[_m]
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != SOURCE_ROOT]
    try:
        import hgevolve
    except ImportError:
        sys.path.insert(0, SOURCE_ROOT)
        import hgevolve


@pytest.fixture
def child_env():
    """The environment for a child Python process: PYTHONPATH starts with the directory this process
    imported hgevolve from (the build tree, or the installed package's site directory)."""
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.abspath(hgevolve.__file__)))
    env["PYTHONPATH"] = os.pathsep.join([root] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    return env
