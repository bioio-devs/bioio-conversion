#!/bin/bash
# Submit the region-write benchmark job.
# Usage: bash run_write_benchmark.sh

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv activate region-write

SCRIPT=/allen/aics/users/brian.whitney/shard_benchmark.py
LOGS=/allen/aics/users/brian.whitney/shard_benchmark4/logs
mkdir -p "$LOGS"

sbatch \
    --job-name="region_write_bench" \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=12:00:00 \
    --output="${LOGS}/%j_region_write_bench.out" \
    --error="${LOGS}/%j_region_write_bench.err" \
    --wrap="python3 $SCRIPT"

echo "Submitted: region write benchmark"
