"""Where the GPU idles in a training step: Chrome traces from make_profile_script.py's
timeline mode (logs/trace_<version>/step<N>.json). Busy time is the union of kernel,
copy and fill intervals; idle = gaps between them inside the step's GPU span.
Each gap is charged to the kernel that ends it. With CPU events (PROFILE_CPU_AT
steps): a gap whose next kernel was launched after the gap began waited for the
CPU, charged to the launching thread's call chain (phase range, deepest Python
frame in our own files, innermost op); otherwise the kernel was queued in time
(GPU-side wait). Also lists the CPU's synchronizing CUDA calls.

    uv run python experiments/csa2/profile_gaps.py logs/trace_b_exact [...]
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

GPU = {"kernel", "gpu_memcpy", "gpu_memset"}
LAUNCH = {"cuda_runtime", "cuda_driver"}
PHASES = ("forward", "backward", "optimizer")
OWN = ("experiments/csa2/b/", "src/nanogpt/", "src/attention/")  # our frames
SYNCS = ("Synchronize", "cudaMemcpy ", "cudaMemcpy(")  # blocking CUDA calls


def chains(events: list[dict]) -> dict[int, tuple[str, str, str]]:
    """correlation id -> (phase, deepest own Python frame, innermost op) of the
    CPU call that launched it: one sweep per thread with a stack of open ranges."""
    by_thread = defaultdict(list)
    for e in events:
        if e.get("ph") == "X" and e.get("cat") not in GPU and "dur" in e:
            by_thread[(e["pid"], e["tid"])].append(e)
    out = {}
    for rows in by_thread.values():
        rows.sort(key=lambda e: (e["ts"], -e["dur"]))
        stack: list[dict] = []
        for e in rows:
            while stack and stack[-1]["ts"] + stack[-1]["dur"] <= e["ts"]:
                stack.pop()
            if e["cat"] in LAUNCH and "correlation" in e.get("args", {}):
                names = [(s["cat"], s["name"]) for s in stack]
                phase = next((n for _, n in names if n in PHASES), "outside phases")
                own = [
                    n
                    for c, n in names
                    if c == "python_function" and any(o in n for o in OWN)
                ]
                ops = [n for c, n in names if c in ("cpu_op", "user_annotation")]
                key = (phase, own[-1] if own else "-", ops[-1] if ops else e["name"])
                out[e["args"]["correlation"]] = key
            stack.append(e)
    return out


def analyze(path: Path) -> None:
    events = json.loads(path.read_text())["traceEvents"]
    gpu = sorted((e for e in events if e.get("cat") in GPU), key=lambda e: e["ts"])
    start, end = gpu[0]["ts"], max(e["ts"] + e["dur"] for e in gpu)
    gaps, busy, reach = [], 0.0, start  # reach: end of the merged busy interval
    for e in gpu:
        if e["ts"] > reach:
            gaps.append((e["ts"] - reach, e))
        busy += max(0.0, e["ts"] + e["dur"] - max(e["ts"], reach))
        reach = max(reach, e["ts"] + e["dur"])
    idle = sum(g for g, _ in gaps)
    span = (end - start) / 1e3
    print(f"\n{path}: GPU span {span:.1f} ms, busy {busy / 1e3:.1f}, idle")
    print(f"  {idle / 1e3:.1f} ms in {len(gaps)} gaps, {len(gpu)} kernels")
    sizes: Counter[float] = Counter()
    for g, _ in gaps:
        sizes[next(b for b in (10, 100, 1000, 1e9) if g < b)] += g
    print("  idle ms by gap size: " + ", ".join(
        f"{'<' if b < 1e9 else '>='} {b if b < 1e9 else 1000:g} us: {t / 1e3:.1f}"
        for b, t in sorted(sizes.items())))  # fmt: skip
    nexts: Counter[str] = Counter()
    for g, e in gaps:
        nexts[e["name"][:90]] += g
    print("  idle ms by the kernel that ends the gap (top 8):")
    for name, t in nexts.most_common(8):
        print(f"    {t / 1e3:7.2f}  {name}")
    if not any(e.get("cat") in LAUNCH for e in events):
        return
    who = chains(events)
    launched = {
        e["args"]["correlation"]: e
        for e in events
        if e.get("cat") in LAUNCH and "correlation" in e.get("args", {})
    }
    cpu: Counter[tuple[str, str, str]] = Counter()
    queued = 0.0
    for g, e in gaps:
        c = e.get("args", {}).get("correlation")
        if c in launched and launched[c]["ts"] >= e["ts"] - g:
            cpu[who.get(c, ("?", "-", "?"))] += g
        else:
            queued += g
    waited = sum(cpu.values()) / 1e3
    print(
        f"  waiting for the CPU {waited:.1f} ms, queued in time {queued / 1e3:.1f} ms"
    )
    print("  CPU waits by (phase, our frame, op), top 12:")
    for (phase, own, op), t in cpu.most_common(12):
        print(f"    {t / 1e3:7.2f}  {phase} | {own[-70:]} | {op[:60]}")
    syncs: Counter[str] = Counter()
    for e in events:
        if e.get("cat") in LAUNCH and any(s in e["name"] for s in SYNCS):
            syncs[e["name"]] += e["dur"]
    print("  CPU time in synchronizing CUDA calls (ms):",
          {k: round(v / 1e3, 2) for k, v in syncs.most_common(5)})  # fmt: skip


if __name__ == "__main__":
    for folder in sys.argv[1:]:
        steps = sorted(Path(folder).glob("step*.json"), key=lambda p: int(p.stem[4:]))
        for path in steps:
            analyze(path)
