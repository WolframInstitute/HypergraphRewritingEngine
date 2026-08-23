#!/usr/bin/env python3
"""Host facts and per-child resource usage, on both platforms this project measures on.

The measurement path was written against Linux only: `machine()` parsed /proc/cpuinfo and
/proc/meminfo, `physical_cores()` parsed /proc/cpuinfo AGAIN, the GPU sampler hard-coded the WSL
nvidia-smi path, and the peak-RSS table shelled out to `/usr/bin/time -v`. On Windows the first
degrades silently to "unknown, ? GB RAM" -- a provenance line that names no machine -- the second
raises, and the fourth cannot run at all.

ONE implementation of each quantity lives here and both generators call it, rather than a Linux
body and a Windows body per caller.

PEAK RSS CANNOT BE MEASURED FROM A LARGE PARENT, which decides the shape of measure(). At fork
the child's address space is a copy-on-write copy of the parent's, so the child's resident set
momentarily EQUALS the parent's; that high-water is recorded in ru_maxrss and survives exec.
Measured here: sampling_cost_smoke reports 7,604 kB under `/usr/bin/time -v` and 13,672 kB when
forked from this interpreter -- which is exactly this interpreter's OWN ru_maxrss. The number is
the measuring process, not the workload. os.posix_spawn does not avoid it either (13,672).

So Linux keeps `/usr/bin/time -v`: a ~2 MB C launcher is not a legacy dependency here, it is the
requirement. Every peak-RSS figure already in the paper came from it and is unaffected. Windows
cannot have the problem -- CreateProcess does not fork -- so it reads the child's counters
through the API instead. Two branches, because the platforms differ in what is CORRECT, not in
what is convenient. Do not "unify" them onto wait4(): the RSS column would silently double.

WHAT DOES NOT PORT EXACTLY, stated rather than smoothed over: Linux ru_minflt counts MINOR
faults (a frame reclaimed without I/O); Windows PageFaultCount counts ALL faults. They are
different quantities and Usage names them apart, so a table generated on one platform is not
silently compared against the other's.
"""
import ctypes
import os
import re
import platform
import shutil
import subprocess
import sys
import tempfile
import time

IS_WINDOWS = os.name == "nt"


class Usage:
    """What one child process cost. Times in seconds, memory in MB."""

    __slots__ = ("wall", "user", "system", "peak_rss_mb", "minor_faults", "all_faults")

    def __init__(self, wall, user, system, peak_rss_mb, minor_faults=None, all_faults=None):
        self.wall = wall
        self.user = user
        self.system = system
        self.peak_rss_mb = peak_rss_mb
        self.minor_faults = minor_faults    # Linux only
        self.all_faults = all_faults        # Windows only

    @property
    def faults(self):
        """The fault count this platform can report, whichever it is."""
        return self.minor_faults if self.minor_faults is not None else self.all_faults

    @property
    def fault_kind(self):
        return "Minor faults" if self.minor_faults is not None else "Page faults"


# --------------------------------------------------------------------------- machine facts

def _windows_cpu_name():
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        with key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        return "unknown"


def _windows_ram_gb():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return "?"
    return "%.0f" % (st.ullTotalPhys / (1024.0 ** 3))


def cpu_name():
    if IS_WINDOWS:
        return _windows_cpu_name()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def ram_gb():
    if IS_WINDOWS:
        return _windows_ram_gb()
    try:
        with open("/proc/meminfo") as f:
            return "%.0f" % (int(f.readline().split()[1]) / 1048576.0)
    except (OSError, ValueError, IndexError):
        return "?"


def machine():
    """The provenance line's machine string."""
    return "%s, %s GB RAM, %s %s" % (cpu_name(), ram_gb(), platform.system(), platform.release())


def topology_is_virtualised():
    """Whether the CPU topology on offer is a hypervisor's rather than the hardware's.

    It decides whether a core COUNT can be believed. Under WSL /proc/cpuinfo is the flattened
    view the hypervisor presents, and on a hybrid part that view is not the machine: an i9-14900K
    is 8 P-cores plus 16 E-cores, 24 physical and 32 logical, and reports here as 16 cores x 2
    threads. Counting (physical id, core id) pairs then yields 16, which is neither number.
    """
    if IS_WINDOWS:
        return False
    try:
        with open("/proc/sys/kernel/osrelease") as f:
            rel = f.read().lower()
        return "microsoft" in rel or "wsl" in rel
    except OSError:
        return False


def physical_cores():
    """Physical cores on this host, or None when the topology cannot be believed.

    The efficiency figure MARKS this count, so it is a claim about the machine and not a
    convenience: drawing the rule at a number this host does not have is worse than not drawing
    it. A (physical id, core id) pair identifies a core on Linux; hyperthreads share one. Windows
    answers the same question through GetLogicalProcessorInformation, counting the entries that
    describe a core rather than a cache or a package.

    None means the count is unavailable rather than zero, and the caller is expected to say so
    rather than substitute a number. That happens under a hypervisor that flattens topology, and
    on a hybrid CPU the flattened count is wrong in both directions at once -- see
    topology_is_virtualised().
    """
    if topology_is_virtualised():
        return None
    if IS_WINDOWS:
        n = _windows_physical_cores()
        if n:
            return n
    else:
        pairs, phys, core = set(), None, None
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("physical id"):
                        phys = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        core = line.split(":")[1].strip()
                    elif not line.strip() and phys is not None and core is not None:
                        pairs.add((phys, core))
                        phys = core = None
            if phys is not None and core is not None:
                pairs.add((phys, core))
            if pairs:
                return len(pairs)
        except OSError:
            pass
    return os.cpu_count() or 1


def _windows_physical_cores():
    RelationProcessorCore = 0

    class SYSTEM_LOGICAL_PROCESSOR_INFORMATION(ctypes.Structure):
        _fields_ = [("ProcessorMask", ctypes.c_void_p),
                    ("Relationship", ctypes.c_int),
                    ("Reserved", ctypes.c_ulonglong * 2)]

    k32 = ctypes.windll.kernel32
    size = ctypes.c_ulong(0)
    k32.GetLogicalProcessorInformation(None, ctypes.byref(size))
    n = size.value // ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)
    if n <= 0:
        return 0
    buf = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION * n)()
    if not k32.GetLogicalProcessorInformation(ctypes.byref(buf), ctypes.byref(size)):
        return 0
    return sum(1 for e in buf if e.Relationship == RelationProcessorCore)


def nvidia_smi_path():
    """Where nvidia-smi is, or None. WSL keeps it off PATH, which reads as "no GPU"."""
    for cand in ("/usr/lib/wsl/lib/nvidia-smi",
                 r"C:\Windows\System32\nvidia-smi.exe"):
        if os.path.exists(cand):
            return cand
    return shutil.which("nvidia-smi")


# --------------------------------------------------------------------------- measurement

class Measured:
    __slots__ = ("returncode", "stdout", "stderr", "usage")

    def __init__(self, returncode, stdout, stderr, usage):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.usage = usage


def measure(cmd, cwd=None, env=None, timeout=None):
    """Run `cmd` and report what it cost, as the kernel accounted it."""
    e = dict(os.environ)
    if env:
        e.update(env)
    if IS_WINDOWS:
        return _measure_windows(cmd, cwd, e, timeout)
    return _measure_gnu_time(cmd, cwd, e, timeout)


GNU_TIME = "/usr/bin/time"


def _measure_gnu_time(cmd, cwd, env, timeout):
    p = subprocess.run([GNU_TIME, "-v"] + list(cmd), cwd=cwd, env=env,
                       capture_output=True, text=True, timeout=timeout)

    def g(pat, cast=float):
        m = re.search(pat, p.stderr)
        if not m:
            raise RuntimeError("%s -v did not report %r; stderr tail:\n%s"
                               % (GNU_TIME, pat, p.stderr[-500:]))
        return cast(m.group(1))

    # The elapsed field is h:mm:ss.ss OR m:ss.ss depending on duration, so the fields are folded
    # from the right rather than matched against one of the two shapes. The label itself contains
    # colons ("(h:mm:ss or m:ss)"), which is what the greedy .* anchors past.
    wall = 0.0
    for part in g(r"Elapsed \(wall clock\) time.*:\s*([\d:.]+)", str).split(":"):
        wall = wall * 60.0 + float(part)

    usage = Usage(wall,
                  g(r"User time \(seconds\): ([\d.]+)"),
                  g(r"System time \(seconds\): ([\d.]+)"),
                  g(r"Maximum resident set size \(kbytes\): (\d+)") / 1024.0,
                  minor_faults=g(r"Minor \(reclaiming a frame\) page faults: (\d+)", int))
    # GNU time's own stderr is prepended to the child's; the report is its last 23 lines.
    child_stderr = "\n".join(p.stderr.splitlines()[:-23])
    return Measured(p.returncode, p.stdout, child_stderr, usage)


def _measure_windows(cmd, cwd, env, timeout):
    """Output through temporary FILES rather than pipes: the counters are read from the child at
    exit, which means reaping it here rather than letting communicate() do it, and a pipe that
    filled would deadlock a child nobody is draining."""
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        t0 = time.perf_counter()
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=err)
        p.wait(timeout=timeout)
        usage = _windows_usage(p)
        usage.wall = time.perf_counter() - t0
        out.seek(0)
        err.seek(0)
        return Measured(p.returncode,
                        out.read().decode("utf-8", "replace"),
                        err.read().decode("utf-8", "replace"),
                        usage)


def _windows_usage(p):
    class FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", ctypes.c_ulong), ("dwHighDateTime", ctypes.c_ulong)]

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    def secs(ft):
        return ((ft.dwHighDateTime << 32) | ft.dwLowDateTime) / 1e7   # 100-ns units

    # The handle stays valid after exit until Popen closes it, and Windows keeps the accounting
    # with it -- which is why these are read here rather than after the with-block.
    h = int(p._handle)
    creation, exit_, kernel, user = FILETIME(), FILETIME(), FILETIME(), FILETIME()
    ok = ctypes.windll.kernel32.GetProcessTimes(
        ctypes.c_void_p(h), ctypes.byref(creation), ctypes.byref(exit_),
        ctypes.byref(kernel), ctypes.byref(user))
    utime = secs(user) if ok else 0.0
    stime = secs(kernel) if ok else 0.0

    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(pmc)
    try:
        psapi = ctypes.windll.psapi
    except (AttributeError, OSError):
        psapi = ctypes.windll.kernel32
    ok = psapi.GetProcessMemoryInfo(ctypes.c_void_p(h), ctypes.byref(pmc), pmc.cb)
    peak_mb = (pmc.PeakWorkingSetSize / (1024.0 * 1024.0)) if ok else 0.0
    faults = pmc.PageFaultCount if ok else 0
    return Usage(0.0, utime, stime, peak_mb, all_faults=faults)


def selftest():
    """The measurement must describe the CHILD, not the process doing the measuring.

    This is the one property that cannot be checked by comparing against another tool, because
    the tool it would be compared against is the one being used. It is checked structurally
    instead: a trivial child's peak RSS must be far below this interpreter's own, which is false
    exactly when the fork high-water is being reported (there the two are EQUAL).
    """
    failures = []

    if not IS_WINDOWS:
        import resource
        own_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        m = measure(["/bin/true"])
        if m.returncode != 0:
            failures.append("/bin/true exited %d" % m.returncode)
        if m.usage.peak_rss_mb > own_mb / 2.0:
            failures.append(
                "peak RSS of /bin/true is %.1f MB against this interpreter's own %.1f MB. A "
                "trivial child cannot cost that; the fork copy-on-write high-water is being "
                "reported instead of the child's own." % (m.usage.peak_rss_mb, own_mb))
        if m.usage.peak_rss_mb <= 0.0:
            failures.append("peak RSS came back as %r" % m.usage.peak_rss_mb)
    else:
        m = measure(["cmd", "/c", "exit", "0"])
        if m.returncode != 0:
            failures.append("cmd /c exit 0 returned %d" % m.returncode)
        if m.usage.peak_rss_mb <= 0.0:
            failures.append("peak RSS came back as %r" % m.usage.peak_rss_mb)

    if cpu_name() == "unknown":
        failures.append("cpu_name() could not name this processor")
    if ram_gb() == "?":
        failures.append("ram_gb() could not size this machine")
    if physical_cores() < 1:
        failures.append("physical_cores() returned %r" % physical_cores())

    for f in failures:
        print("FAIL: " + f)
    if not failures:
        print("OK: %s | %s physical cores | measurement describes the child" %
              (machine(), physical_cores()))
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(selftest())
    print("machine        :", machine())
    print("physical cores :", physical_cores())
    print("nvidia-smi     :", nvidia_smi_path())
    if len(sys.argv) > 1:
        m = measure(sys.argv[1:])
        u = m.usage
        print("rc=%d wall=%.3fs user=%.3fs sys=%.3fs peak=%.1fMB %s=%s"
              % (m.returncode, u.wall, u.user, u.system, u.peak_rss_mb, u.fault_kind, u.faults))
