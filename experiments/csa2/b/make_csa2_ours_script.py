"""Writes train_csa2_ours[_VARIANT].py next to this file: train_csa2_ref.py (the V4.1
reference port, make_csa2_ref_script.py) with scratch src/nanogpt/csa2_ours.py's
layers: the same model picking by our summed query (env CSA2_SELECT: exact or
tournament) instead of its learned indexer. Variants change the Config (*_smooth
also the schedule):
    (none)  csa2_ref.SMALL: ratio 2 at layers 1-4, ratio 1 at layers 5-9
    r4      ratio 4 at layers 1-9 (one entry per 4 tokens; sources and picks kept)
    r4_smooth  r4 with the stage-switch fix: the learning rate ramps linearly to each
            stage's lr_mul over 3% of the schedule (300 of 10000 steps) instead of
            jumping (+52% at 1/3, +13% at 2/3 inside the cooldown); the mean
            train loss per token of every 50 steps (MTP- and prefix-weighted) is
            logged, to place any stall
    smooth  csa2_ref.SMALL with r4_smooth's schedule fix (base of train_csa2_pf_smooth.py)
    b/run_b.sh exact train_csa2_ours.py
"""

# ruff: noqa: E501  (patch strings must match the source lines)
import sys
from pathlib import Path

HERE = Path(__file__).parent
CONFIGS = {
    "": "csa2_ref.SMALL",
    "r4": "csa2_ref.Config(compress_ratios=(0,) + (4,) * 9)",
    "r4_smooth": "csa2_ref.Config(compress_ratios=(0,) + (4,) * 9)",
    "smooth": "csa2_ref.SMALL",
}
SMOOTH = [
    (
        """        stage, _ = self.lookup(step)
        lr = stage.lr_mul
""",
        """        stage, _ = self.lookup(step)
        lr = stage.lr_mul
        # stage-switch fix: ramp from the previous stage's lr_mul instead of a jump
        ramp = max(1, round(0.03 * self.scheduled_iterations))
        for i, (start, _end) in enumerate(self.boundaries[1:-1], 1):
            if start <= step < start + ramp:
                prev = self.stages[i - 1].lr_mul
                lr = prev + (lr - prev) * (step - start) / ramp
""",
    ),
    (
        """for step in range(train_steps + 1):
    last_step = (step == train_steps)
""",
        """train_loss = torch.zeros((), device=device)  # sum since the last log
for step in range(train_steps + 1):
    last_step = (step == train_steps)
""",
    ),
    (
        """        inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = train_loader.send(training_manager.train_loader_send_args)
        training_manager.sparse_index_update(step, bigram_cpu)
        loss = model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args()).sum() * grad_scale
""",
        """        inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = train_loader.send(training_manager.train_loader_send_args)
        training_manager.sparse_index_update(step, bigram_cpu)
        loss = model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args()).sum() * grad_scale
        train_loss += loss.detach()
""",
    ),
    (
        """print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)
""",
        """print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)
    if (step + 1) % 50 == 0:
        print0(f"step:{step+1}/{train_steps} train_loss:{train_loss.item() / 50 / (training_manager.batch_size * grad_scale):.4f} lr:{training_schedule.get_lr(step):.4f}", console=True)
        train_loss.zero_()
""",
    ),
]
variant = sys.argv[1] if len(sys.argv) > 1 else ""
src = (HERE / "train_csa2_ref.py").read_text()
edits = [
    (
        """from nanogpt import csa2_flex, csa2_ref
for _module in (csa2_ref, csa2_flex):""",
        """from nanogpt import csa2_flex, csa2_ours, csa2_ref, csa2_select_fast
for _module in (csa2_ref, csa2_flex, csa2_ours, csa2_select_fast):""",
    ),
    (
        """print(f"CSA2 reference layers, FP4 caches {csa2_ref.FP4}", flush=True)""",
        """print(f"CSA2 reference layers, our picks {csa2_ours.RULE.__name__}, FP4 caches {csa2_ref.FP4}", flush=True)""",
    ),
    (
        """        config = csa2_ref.SMALL
        self.csa2 = nn.ModuleList(csa2_ref.Csa2RefAttention(i, config) for i in range(num_attn_layers))""",
        f"""        config = {CONFIGS[variant]}
        self.csa2 = nn.ModuleList(csa2_ours.Csa2OursAttention(i, config) for i in range(num_attn_layers))""",
    ),
]
edits += SMOOTH * variant.endswith("smooth")
for old, new in edits:
    assert src.count(old) == 1, f"not found once: {old[:80]!r}"
    src = src.replace(old, new)
out = HERE / f"train_csa2_ours{'_' * bool(variant)}{variant}.py"
out.write_text(src)
print("wrote", out)
