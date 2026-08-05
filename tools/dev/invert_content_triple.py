#!/usr/bin/env python3
"""Recover the (from_class, to_class, rule) behind a reconstructed event identity.

hgcommon::qr_content_hash is FNV-1a over three words:

    s = OFFSET
    s = (s ^ from_class) * PRIME
    s = (s ^ to_class)   * PRIME
    s = (s ^ rule)       * PRIME

PRIME is odd, so multiplication is invertible mod 2^64 and the whole mixing is a
bijection that can be run backwards. Given a final hash and a guess for `rule`,
the pair (from_class, to_class) is constrained to a one-parameter family: fix
either and the other is determined. That is enough to test the hypotheses worth
testing -- a degenerate zero component, or the two engines feeding the same
class hash in a different position.

Used on the #124 divergence, where CPU and GPU produce the same NUMBER of
distinct reconstructed identities with ZERO overlap.
"""

MASK = (1 << 64) - 1
OFFSET = 14695981039346656037
PRIME = 1099511628211


def pow_inv(p):
    """Modular inverse of an odd multiplier mod 2^64, by Newton iteration."""
    inv = 1
    for _ in range(6):                      # doubles correct bits each round: 1,2,4,...,64
        inv = (inv * (2 - p * inv)) & MASK
    return inv


PINV = pow_inv(PRIME)


def content_hash(a, b, r):
    s = OFFSET
    s = ((s ^ a) * PRIME) & MASK
    s = ((s ^ b) * PRIME) & MASK
    s = ((s ^ r) * PRIME) & MASK
    return s


def recover(final, rule, known_b=None, known_a=None):
    """Given the hash and rule, return (a, b) with one of them supplied."""
    s2 = ((final * PINV) & MASK) ^ rule
    t1 = (s2 * PINV) & MASK                 # == s1 ^ b
    if known_b is not None:
        s1 = t1 ^ known_b
        a = ((s1 * PINV) & MASK) ^ OFFSET
        return a, known_b
    if known_a is not None:
        s1 = (((known_a ^ OFFSET) * PRIME) & MASK)
        b = t1 ^ s1
        return known_a, b
    return None, None



def prove_basis_defect():
    """The digit-dropped basis, applied to the triple the HOST resolved to, reproduces the
    value the DEVICE reported. That is the whole of #124: same prime, same order, same
    inputs, one wrong starting constant.

    Triples come from the endpoint diagnostic in gpu_differential_tests, which resolved each
    CPU identity by searching the run's real canonical class hashes.
    """
    BAD = OFFSET // 10          # 1469598103934665603 -- the basis missing its last digit

    def bad_hash(a, b, r):
        s = BAD
        s = ((s ^ a) * PRIME) & MASK
        s = ((s ^ b) * PRIME) & MASK
        s = ((s ^ r) * PRIME) & MASK
        return s

    print(f'\ncorrect basis {OFFSET}, device basis {BAD}, '
          f'device == correct//10: {BAD == OFFSET // 10}')

    # ONLY a workload whose reconstruction has a SINGLE distinct identity per engine can be
    # checked this way. The diagnostic reports the smallest identity of each side's set, and
    # "smallest of the CPU set" and "smallest of the GPU set" name the same EVENT only when each
    # set is a singleton. deep_cone_reduction_d6 is: 248 endpoints, 1 distinct identity on each
    # side. Workloads with 9 and 34 distinct identities were tried here first and their rows
    # tested nothing, because the two smallest values are simply different events.
    #
    # (workload, from_class, to_class, rule, cpu identity, gpu identity)
    singleton = [
        ('deep_cone_reduction_d6',
         8973042842819422521, 8973042842819422521, 0,
         8836476779998324405, 11458423332341610703),
    ]
    ok = True
    for name, a, b, r, cpu_v, gpu_v in singleton:
        good = content_hash(a, b, r)
        bad = bad_hash(a, b, r)
        good_ok, bad_ok = good == cpu_v, bad == gpu_v
        ok = ok and good_ok and bad_ok
        print(f'  {name}  (1 distinct identity per engine, so the pairing is forced)')
        print(f'    correct basis -> {good:20d}  cpu reported {cpu_v:20d}  {"MATCH" if good_ok else "no"}')
        print(f'    device  basis -> {bad:20d}  gpu reported {gpu_v:20d}  {"MATCH" if bad_ok else "no"}')
    print(f'\n  #124 mechanism {"PROVEN" if ok else "NOT reproduced -- the diagnosis is wrong"}: '
          'one wrong FNV basis, nothing else. A 64-bit agreement on both engines from the same\n'
          '  triple is not a coincidence, and it is reached with the same prime, the same field\n'
          '  order and the same inputs -- only the starting constant differs.')
    return ok


def main():
    # Measured 2026-08-05 by the endpoint diagnostic in gpu_differential_tests.
    cases = [
        ('deep_cone_reduction_d6      cpu', 8836476779998324405),
        ('deep_cone_reduction_d6      gpu', 11458423332341610703),
        ('multi_initial_iso_roots_kept cpu', 1624003546513094352),
        ('multi_initial_iso_roots_kept gpu', 753859263396775162),
    ]

    prove_basis_defect()

    print('sanity: the inverse really inverts')
    probe = content_hash(0xDEADBEEF, 0x1234, 3)
    a, b = recover(probe, 3, known_b=0x1234)
    print(f'  round trip from_class {a:#x} (expected 0xdeadbeef), to_class {b:#x}')
    assert a == 0xDEADBEEF, 'inverse is wrong; nothing below means anything'

    print('\nis either value the degenerate triple (0, 0, rule)?')
    for r in range(8):
        h = content_hash(0, 0, r)
        for name, v in cases:
            if h == v:
                print(f'  MATCH {name}: qr_content_hash(0, 0, {r})')
    print('  (no line above means no case is the all-zero triple)')

    print('\nassuming to_class == 0, what from_class would each imply, per rule?')
    for name, v in cases:
        for r in range(3):
            a, _ = recover(v, r, known_b=0)
            print(f'  {name}  rule={r}  from_class={a:#018x}')

    print('\nassuming from_class == to_class (a self-transition), the implied class:')
    for name, v in cases:
        for r in range(3):
            # a == b means s1 = t1 ^ a and a = (s1 * PINV) ^ OFFSET; solve by search over
            # the low bits is not needed -- state the constraint and let the caller compare
            # against the run's real canonical hashes.
            s2 = ((v * PINV) & MASK) ^ r
            t1 = (s2 * PINV) & MASK
            print(f'  {name}  rule={r}  s1^to_class={t1:#018x}')


if __name__ == '__main__':
    main()
