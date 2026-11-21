# Colab T4 draft results (P4: Lina)

Exploratory runs on Google Colab (Tesla T4, 2 shared CPU cores), **not** the official numbers.
The official results come from P1's RTX 3050 laptop (`results/` at the top level).

| Folder | Assignment | Notebook entry |
|---|---|---|
| `baseline_colab/` | baseline (reference CNN) on the Colab GPU, reference for A6.9 | A6.9 |
| `exp_*/` | the 8 single-change redesign experiments + seed control | A6.9 and its follow-up |
| `torch_profiler_baseline/` | GPU profile of the baseline pipeline (GPU idle 95%) | A7.2–A7.3 |

They are the evidence for the A6.9 conclusions (ResNet-18 + lr 0.0001 is clearly better; the other
changes are within run-to-run variation) and for the draft GPU profile. Epoch times are comparable
between these runs (same Colab setup), but not with the laptop's.
