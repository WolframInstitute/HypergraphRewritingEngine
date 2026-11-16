"""Every script in examples/ runs to completion."""
import os
import subprocess
import sys

import pytest

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
SCRIPTS = sorted(f for f in os.listdir(EXAMPLES) if f.endswith(".py"))


@pytest.mark.parametrize("script", SCRIPTS)
def test_example_runs(script, child_env):
    env = child_env
    p = subprocess.run([sys.executable, os.path.join(EXAMPLES, script)], env=env,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    assert p.stdout.strip()
