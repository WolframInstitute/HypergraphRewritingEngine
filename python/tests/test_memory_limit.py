"""A job over its "MemoryLimit" returns a partial result with the "memory exhausted" warning in the
Python process, as hg_evolve does (hypergraph::JobMemoryLimit applies in run_job)."""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))

# The child processes read their resident set from /proc/self/status and cap their address space with
# resource.RLIMIT_AS.
pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self/status (Linux)")


def test_a_job_over_its_memory_limit_returns_a_partial_result(child_env):
    env = child_env
    p = subprocess.run([sys.executable, os.path.join(HERE, "memory_limit_child.py")], env=env,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["num_states"] > 100
    assert out["memory_limit"] == 64 << 20
    assert any("memory exhausted" in w and "TRUNCATED" in w for w in out["warnings"]), out


def test_calls_in_one_process_return_their_memory(child_env):
    """Each call releases what it built: the 6-step job gives the same count before and after
    three jobs that fill a 64 MB "MemoryLimit", and the resident set after the last call is within
    16 MB of its value after the first."""
    env = child_env
    p = subprocess.run([sys.executable, os.path.join(HERE, "repeated_calls_child.py")], env=env,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    calls = json.loads(p.stdout.strip().splitlines()[-1])
    assert [c["truncated"] for c in calls] == [False, True, True, True, False], calls
    assert calls[0]["states"] == calls[-1]["states"], calls
    assert calls[-1]["rss_kb"] - calls[0]["rss_kb"] < 16 * 1024, calls
