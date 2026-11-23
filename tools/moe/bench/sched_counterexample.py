#!/usr/bin/env python3
# Independent re-derivation of the shared-bandwidth counterexample.
# Different implementation from the original: continuous-time event simulation
# (advance to the next job completion), exhaustive over all DISTINCT orderings,
# and also reports how many orderings beat cold-first. Verifies:
#   (a) cold-first (longest-first) is NOT makespan-optimal even in a pure
#       shared-bandwidth model with no copy cost;
#   (b) the optimum equals the lower bound sum(work)/Bmax.
from itertools import permutations

def simulate(order, curves):
    """curves[n] = aggregate bandwidth with n concurrent readers.
    Greedy: admit up to 2 jobs; bandwidth shared equally among active readers;
    run to the next completion; admit more. Returns makespan."""
    order = list(order)
    active = []          # remaining work per active job
    t = 0.0
    while order or active:
        # admission: fill up to 2 slots from the head of the queue
        while order and len(active) < 2:
            active.append(float(order.pop(0)))
        n = len(active)
        per = curves[n] / n            # equal share per active reader
        dt = min(a / per for a in active)
        t += dt
        active = [a - per * dt for a in active]
        active = [a for a in active if a > 1e-9]
    return t

def main():
    curves = {1: 1.0, 2: 1.5}          # B(1)=1, B(2)=1.5, equal sharing
    jobs = (3, 3, 2, 2, 2)
    lower = sum(jobs) / max(curves.values())

    # cold-first / longest-first
    cold = tuple(sorted(jobs, reverse=True))
    t_cold = simulate(cold, curves)

    # exhaustive over distinct permutations
    best_t, best_p = None, None
    n_beat = 0
    seen = set()
    for p in permutations(jobs):
        if p in seen:
            continue
        seen.add(p)
        t = simulate(p, curves)
        if best_t is None or t < best_t - 1e-9:
            best_t, best_p = t, p
        if t < t_cold - 1e-9:
            n_beat += 1

    print("shared-channel model: W=2, B(1)=%.1f, B(2)=%.1f, equal sharing" %
          (curves[1], curves[2]))
    print("cold-first order  : %s  makespan = %.4f" % (cold, t_cold))
    print("optimal (exhaust.) : %s  makespan = %.4f" % (best_p, best_t))
    print("lower bound sum/B  : %.4f" % lower)
    print("distinct orders    : %d" % len(seen))
    print("orders beating cold-first: %d" % n_beat)
    assert abs(best_t - lower) < 1e-9, "optimum should hit the lower bound"
    assert t_cold > best_t + 1e-9, "cold-first should NOT be optimal"
    assert n_beat > 0, "other orders must beat cold-first"
    print("VERIFIED: cold-first is not optimal; a strictly better order exists.")

if __name__ == "__main__":
    main()
