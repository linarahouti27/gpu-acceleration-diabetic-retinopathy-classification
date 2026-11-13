"""A6.6: parameters and FLOPs per 224x224 image, reference CNN vs ResNet-18.

Runs on the CPU (no GPU needed). FlopCounterMode counts 2 FLOPs per multiply-add.
Writes results/model_size.csv.

    python benchmarks/model_size.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode
from src.model import build_model

rows = []
for name in ["reference_cnn", "resnet18"]:
    model = build_model(name, num_classes=5).eval()
    n_params = sum([t.numel() for t in model.parameters()])   # total number of learned numbers

    x = torch.zeros(1, 3, 224, 224)                   # one fake image (the values don't matter)
    with FlopCounterMode(display=False) as counter:   # PyTorch counts every operation inside
        with torch.no_grad():
            model(x)
    flops = counter.get_total_flops()                 # FLOPs for one image

    rows.append({"model": name,
                 "params_millions": n_params / 1e6,
                 "gflops_per_image": flops / 1e9})
    print(rows[-1])

os.makedirs("results", exist_ok=True)
pd.DataFrame(rows).to_csv("results/model_size.csv", index=False)
print("-> results/model_size.csv")
