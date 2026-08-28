#!/usr/bin/env python3
"""Name the events a disagreeing run lacks, given two HG_DET_DUMP listings.

    det_dump_diff.py <reference.txt> <disagreeing.txt>

An event is identified by (input canonical hash, output canonical hash, rule); raw ids differ
between runs and are not compared. For each event only the reference has, the reference's own
listing says whether the match that produced it was a DELTA match (it consumed an edge that the
event creating its input state produced) or a FORWARDED one (every consumed edge predates that
state), which is the split between the two paths a match reaches a state by.
"""
import sys
from collections import Counter, defaultdict

def load(path):
    states, events, kept, counts = {}, [], set(), {}
    for line in open(path):
        p = line.split()
        if not p: continue
        if p[0] == 'S':
            states[int(p[1])] = int(p[2])
            if 'M' in p:
                m = p.index('M'); counts[int(p[1])] = tuple(int(x) for x in p[m+1:m+4])
        elif p[0] == 'E':
            i = p.index('C'); j = p.index('P')
            events.append(dict(id=int(p[1]), in_raw=int(p[2]), out_raw=int(p[3]), in_c=int(p[4]),
                               out_c=int(p[5]), rule=int(p[6]),
                               consumed=[int(x) for x in p[i+1:j]], produced=[int(x) for x in p[j+1:]]))
        elif p[0] == 'K': kept.add((int(p[1]), int(p[2])))
    return states, events, kept, counts

ref_s, ref_e, ref_k, ref_m = load(sys.argv[1])
oth_s, oth_e, oth_k, oth_m = load(sys.argv[2])
key = lambda e: (e['in_c'], e['out_c'], e['rule'])
rc, oc = Counter(map(key, ref_e)), Counter(map(key, oth_e))
print(f"reference: {len(ref_s)} states {len(ref_e)} events {len(ref_k)} causal; other: {len(oth_s)} states {len(oth_e)} events {len(oth_k)} causal")
creator = {e['out_raw']: e for e in ref_e}          # raw state -> the event that created it
producer_of = {}
for e in ref_e:
    for x in e['produced']: producer_of[x] = e['id']
missing = rc - oc
extra = oc - rc
print(f"missing from other: {sum(missing.values())}; extra in other: {sum(extra.values())}")
for k, n in missing.items():
    for e in ref_e:
        if key(e) != k: continue
        c = creator.get(e['in_raw'])
        fresh = [x for x in e['consumed'] if c and x in c['produced']]
        kind = 'DELTA (consumes an edge its input state was created with)' if fresh else 'FORWARDED (every consumed edge predates its input state)'
        prods = sorted({producer_of.get(x, -1) for x in e['consumed']})
        depth, cur = 0, e['in_raw']
        while cur in creator: cur = creator[cur]['in_raw']; depth += 1
        # events at the same input state in both runs, and the twins consuming the same edge set
        same_in_ref = [f for f in ref_e if f['in_c'] == e['in_c'] and f['rule'] == e['rule']]
        same_in_oth = [f for f in oth_e if f['in_c'] == e['in_c'] and f['rule'] == e['rule']]
        twins_ref = [f for f in same_in_ref if set(f['consumed']) == set(e['consumed'])]
        oth_by_key = Counter(map(key, same_in_oth))
        twins_present = sum(oth_by_key[key(f)] for f in twins_ref)
        fresh_consumers_ref = [f for f in same_in_ref if c and any(x in c['produced'] for x in f['consumed'])]
        fresh_present = sum(oth_by_key[key(f)] for f in fresh_consumers_ref)
        print(f"  event {e['id']} rule {e['rule']} in={e['in_raw']}(c{e['in_c']}) depth={depth} out={e['out_raw']}(c{e['out_c']}) consumed={e['consumed']} produced-by={prods} -> {kind}")
        if c: print(f"    input state {e['in_raw']} created by event {c['id']} consuming {c['consumed']} producing {c['produced']}")
        # the counterpart of the input state in the other run: same canonical hash, same creating event, same depth
        oth_creator = {f['out_raw']: f for f in oth_e}
        def depth_of(raw, cr):
            d = 0
            while raw in cr: raw = cr[raw]['in_raw']; d += 1
            return d
        cands = [r for r, h in oth_s.items() if h == e['in_c'] and (r in oth_creator) == (c is not None)
                 and (c is None or key(oth_creator[r]) == key(c)) and depth_of(r, oth_creator) == depth]
        print(f"    match-task counts (pushed, completed, matches) at input state: ref {ref_m.get(e['in_raw'])}; counterparts in other: {[(r, oth_m.get(r)) for r in cands]}")
        print(f"    events at this input state: ref {len(same_in_ref)} other {len(same_in_oth)}; twins over the same edge set: ref {len(twins_ref)} other {twins_present}; delta events (consume a fresh edge): ref {len(fresh_consumers_ref)} other {fresh_present}")
        break
