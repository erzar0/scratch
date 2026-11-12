"""Writes b/<script>_profile.py: a record 91 training script (B scripts, or r91's
train_csa2.py) that runs torch.profiler over the training steps in env PROFILE_AT,
each micro-batch's forward and backward and the optimizer step in record_function
ranges (synchronized on profiled steps only), and writes every CUDA kernel's time
with its phase to env PROFILE_OUT, plus the wall time of each phase and of the
whole step (phase "step", kernel "(wall)"), then exits. b/ holds triton_kernels.py
copies, so the script runs from there.
Timeline mode (env PROFILE_TRACE = a directory): no syncs inside the step, each
profiled step saved as a Chrome trace <dir>/step<N>.json; steps in PROFILE_CPU_AT
record CPU ops and Python stacks, the rest GPU kernels only (no CPU slowdown).

    python make_profile_script.py b/train_tournament_kv_fast.py
"""

# ruff: noqa: E501  (patch strings must match the training scripts' lines)
import sys
from pathlib import Path

HELPERS = '''# profiler (make_profile_script.py): kernels of steps PROFILE_AT by phase
import contextlib
import csv
from torch.autograd import DeviceType
PROFILE_AT = sorted(int(s) for s in os.environ["PROFILE_AT"].split(","))
PHASES = ("forward", "backward", "optimizer")
TRACE = os.environ.get("PROFILE_TRACE")
CPU_AT = {int(s) for s in os.environ.get("PROFILE_CPU_AT", "").split(",") if s}
_rows = []


def _profile_start(step):
    if step not in PROFILE_AT:
        return None
    torch.cuda.synchronize()
    cpu = not TRACE or step in CPU_AT
    activities = [torch.profiler.ProfilerActivity.CUDA]
    activities += [torch.profiler.ProfilerActivity.CPU] if cpu else []
    prof = torch.profiler.profile(activities=activities, with_stack=bool(TRACE) and cpu)
    prof.__enter__()
    prof.whole = torch.profiler.record_function("step")
    prof.whole.__enter__()
    return prof


@contextlib.contextmanager
def _phase(prof, name):
    if prof is None:
        yield
        return
    with torch.profiler.record_function(name):
        yield
        if not TRACE:
            torch.cuda.synchronize()


def _profile_stop(prof, step):
    if prof is None:
        return
    torch.cuda.synchronize()
    prof.whole.__exit__(None, None, None)
    prof.__exit__(None, None, None)
    if TRACE:
        prof.export_chrome_trace(f"{TRACE}/step{step}.json")
        print0(f"trace written: step {step}", console=True)
        if step == PROFILE_AT[-1]:
            os._exit(0)
        return
    events = prof.events()
    spans = [e for e in events if e.name in PHASES and e.device_type == DeviceType.CPU]
    for e in events:
        if e.name in (*PHASES, "step") and e.device_type == DeviceType.CPU:
            _rows.append((step, e.name, "(wall)", e.time_range.elapsed_us()))
    for e in events:
        if e.device_type != DeviceType.CUDA or e.name in (*PHASES, "step"):
            continue
        t = e.time_range.start
        phase = next((s.name for s in spans if s.time_range.start <= t <= s.time_range.end), "other")
        _rows.append((step, phase, e.name, e.time_range.elapsed_us()))
    print0(f"profiled step {step}", console=True)
    if step == PROFILE_AT[-1]:
        with open(os.environ["PROFILE_OUT"], "w", newline="") as f:
            out = csv.writer(f)
            out.writerow(("step", "phase", "kernel", "us"))
            out.writerows(_rows)
        print0(f"profile written: {os.environ['PROFILE_OUT']}", console=True)
        os._exit(0)


'''

src_path = Path(sys.argv[1])
src = src_path.read_text()
anchor = "train_steps = training_schedule.total_steps\n"
assert src.count(anchor) == 1
src = src.replace(anchor, HELPERS + anchor)
start = src.index("    # --------------- TRAINING SECTION -----------------\n")
end = src.index("    # logging\n", start)
block = src[start:end]
new = block
for old, repl in (
    (
        "    for idx in range(grad_accum_steps):\n",
        "    _prof = _profile_start(step)\n    for idx in range(grad_accum_steps):\n",
    ),
    (
        "        loss = model(",
        "        with _phase(_prof, \"forward\"):\n            loss = model(",
    ),
    (
        "        loss.backward()\n",
        "        with _phase(_prof, \"backward\"):\n            loss.backward()\n",
    ),
    (
        "    training_manager.step_optimizers(step)\n    model.quantize_attn_fp8()\n    model.quantize_mlp_fp8()\n",
        "    with _phase(_prof, \"optimizer\"):\n        training_manager.step_optimizers(step)\n        model.quantize_attn_fp8()\n        model.quantize_mlp_fp8()\n    _profile_stop(_prof, step)\n",
    ),
):
    assert new.count(old) == 1, old
    new = new.replace(old, repl)
src = src[:start] + new + src[end:]
out = Path(__file__).parent / "b" / (src_path.stem + "_profile.py")
out.write_text(src)
print("wrote", out)
