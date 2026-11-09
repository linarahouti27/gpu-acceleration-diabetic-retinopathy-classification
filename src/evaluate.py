"""Model evaluation.

Computes accuracy, macro-precision, macro-recall, macro-F1, per-class results and the
confusion matrix. Final evaluation is on the held-out test set only.

Usage: python -m src.evaluate --config <config> --checkpoint <checkpoint.pt>
"""

import argparse


def evaluate(model, dataloader, device):
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
