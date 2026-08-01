# Does each quotient row's fingerprint equal its full-capture twin's?
#
# Run from the repository root:  python3 tools/golden_quotient_twin_check.py
#
# SPEC 5.4 requires quotient's observable output to be IDENTICAL to full capture. If that holds
# row for row, then every quotient row inherits the independent check its full-capture twin
# already has, and the 51 "pin" rows stop being cells with no oracle behind them.
import sys, collections
rows = {}
for line in open("reference/golden_matrix.txt"):
    if line.startswith("#") or not line.strip():
        continue
    p = line.split()
    if len(p) < 12:
        continue
    case, sm, em, quot = p[0], p[1], p[2], p[3]
    prov, fp = p[-1], p[-2]
    # remaining numeric columns, excluding the fingerprint
    nums = p[4:-2]
    rows[(case, sm, em, quot)] = (nums, fp, prov)

same_fp = diff_fp = same_counts = missing = 0
examples = []
for (case, sm, em, quot), (nums, fp, prov) in rows.items():
    if quot != "1":
        continue
    twin = rows.get((case, sm, em, "0"))
    if twin is None:
        missing += 1
        continue
    tnums, tfp, tprov = twin
    if fp == tfp:
        same_fp += 1
    else:
        diff_fp += 1
        if len(examples) < 6:
            examples.append((case, sm, em, nums, tnums))
    if nums == tnums:
        same_counts += 1

print(f"quotient rows            : {same_fp + diff_fp + missing}")
print(f"  fingerprint == full    : {same_fp}")
print(f"  fingerprint != full    : {diff_fp}")
print(f"  all count cols == full : {same_counts}")
print(f"  no full-capture twin   : {missing}")
if examples:
    print("\nfirst differing rows (quotient counts vs full counts):")
    for c, sm, em, a, b in examples:
        print(f"  {c:<22} {sm:<10} {em:<10} q={a}  full={b}")

# A gate (ctest: quotient_twin_check): SPEC 5.4 requires every quotient row to equal its
# full-capture twin. Fingerprint drift, any count-column drift, and a quotient row with no twin
# all fail -- a missing twin would otherwise let a row pass by never being compared.
total_q = same_fp + diff_fp + missing
sys.exit(1 if (diff_fp or missing or same_counts < total_q - missing) else 0)
