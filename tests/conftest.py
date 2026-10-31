"""Test setup: CPU only, so the GPU stays free for training runs."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
