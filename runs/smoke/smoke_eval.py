"""Throwaway GPU smoke of every qwen.evaluate study and selector; results here.

uv run python runs/smoke/smoke_eval.py <stage> <step>...  steps: base csa2 sel route short
"""

import logging
import os
import sys
import time
from pathlib import Path

import torch

from qwen import evaluate, route
from qwen import train as qt

OUT = Path(__file__).parent
evaluate.OUT = OUT
route.PATH = OUT / "router.pt"
stage, steps = sys.argv[1], sys.argv[2:]
qt.final_stage = lambda _: stage
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
torch.set_float32_matmul_precision("high")
clock = time.time()


def peak(label: str) -> None:
    global clock
    gib = torch.cuda.max_memory_allocated() / 2**30
    logging.info("PEAK %s %.2f GiB, %.0f s", label, gib, time.time() - clock)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    clock = time.time()


os.environ["TASKS"] = "niah_single_1,babilong_qa1,icl_trec_coarse"
if "base" in steps:
    os.environ["LIMIT"] = "4"
    evaluate.lm("qwen")
    peak("qwen lm")
    os.environ["LIMIT"] = "1"
    evaluate.long_tasks("qwen")
    peak("qwen long")
    os.environ["LIMIT"] = "2"
    evaluate.generation_check("qwen")
    peak("qwen check")
if "csa2" in steps:
    os.environ["LIMIT"] = "4"
    evaluate.lm("csa2")
    peak("csa2 lm")
    os.environ["LIMIT"] = "1"
    evaluate.long_tasks("csa2")
    peak("csa2 long")
if "sel" in steps:
    os.environ["LIMIT"] = "4"
    for row in ("csa2:window", "csa2:random", "csa2:oracle", "csa2:mean", "csa2:bound"):
        evaluate.lm(row)
        peak(f"{row} lm")
if "route" in steps:
    os.environ["STEPS"] = "3"
    route.main()
    peak("route train")
    os.environ["LIMIT"] = "1"
    for row in ("csa2:learned", "csa2:mixed"):
        evaluate.lm(row)
        peak(f"{row} lm")
if "short" in steps:
    os.environ["LIMIT"] = "1"
    evaluate.short("qwen")
    peak("qwen short")
    evaluate.short("csa2")
    peak("csa2 short")
