#!/usr/bin/env python3
"""Strict CPU-only production-token preflight before accelerator allocation."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from picoagent.training.config import TrainingConfig  # noqa: E402
from picoagent.training.prepared import load_prepared_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    prepared = load_prepared_dataset(TrainingConfig.load(args.config))
    print(json.dumps({'prepared_admission_verified': True,
                      'manifest': str(prepared.path),
                      'identity': prepared.identity,
                      'accelerator_allocated': False}, sort_keys=True))


if __name__ == '__main__':
    main()
