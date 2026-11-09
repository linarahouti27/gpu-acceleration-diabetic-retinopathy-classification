"""Stage-level CPU vs GPU benchmarks.

Rules: warm up the GPU, call torch.cuda.synchronize() around timed regions, repeat
runs and report mean ± std, distinguish cold-start from steady-state.
Writes results to results/timing.csv.
"""


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
