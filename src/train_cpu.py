"""Module D + E (CPU reference): Augmentation and CNN training.

Augmentation (flips, rotations) is applied to training images only. Fixed seeds,
logging and checkpoints are required for reproducibility.

Usage: python -m src.train_cpu --config configs/baseline.yaml
"""

import argparse


def train(config):
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
