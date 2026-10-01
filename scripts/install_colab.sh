#!/usr/bin/env bash
set -euo pipefail
# Official Google package: https://github.com/googlecolab/google-colab-cli
uv tool install 'google-colab-cli==0.7.4'
colab version
# Authentication and compute allocation are separate, deliberate operator steps.

