"""A sweep in parallel across processes. Each worker runs one evolution at a time with one engine
thread, under the engine's memory limit (tuning "MemoryLimit"), so an evolution that outgrows it is
truncated with a warning instead of filling memory, and an engine crash ends that worker only (the
pool raises BrokenProcessPool)."""
import concurrent.futures
import multiprocessing

LIMIT_BYTES = 2 << 30  # 2 GB per evolution


def run(job):
    import hgevolve as hg
    seed, rate = job
    rule = ([[1, 1]], [[1, 1], [1, 1]])
    r = hg.evolve(rule, [[1, 1], [1, 1]], 6, data=["num_states", "num_events"],
                  transition_rate=rate, random_seed=seed, threads=1,
                  tuning={"MemoryLimit": LIMIT_BYTES})
    return seed, rate, r.num_states, r.num_events


if __name__ == "__main__":
    jobs = [(seed, rate) for seed in range(4) for rate in (0.25, 0.5)]
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=ctx) as pool:
        for seed, rate, states, events in pool.map(run, jobs):
            print(f"seed {seed} rate {rate}: {states} states, {events} events")
