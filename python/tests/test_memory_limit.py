"""A job over its "MemoryLimit" returns a partial result with the "memory exhausted" warning in the
Python process, as hg_evolve does (hypergraph::JobMemoryLimit applies in run_job)."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def test_a_job_over_its_memory_limit_returns_a_partial_result():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.dirname(HERE)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    p = subprocess.run([sys.executable, os.path.join(HERE, "memory_limit_child.py")], env=env,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["num_states"] > 100
    assert out["memory_limit"] == 64 << 20
    assert any("memory exhausted" in w and "TRUNCATED" in w for w in out["warnings"]), out
