
export OUT_ROOT="/allen/aics/users/brian.whitney/benchmark" # OUTPUT PATH
export BENCH_PY="/home/brian.whitney/benchmark.py"   # adjust to your benchmark
export DATA_SOURCE="/home/brian.whitney/data_source.json" # your datasource

mkdir -p ~/logs

# Count how many runs are in the data source
NUM_RUNS=$(python - <<PY
import json
with open("${DATA_SOURCE}") as f:
    print(len(json.load(f).get("runs", [])))
PY
)

for i in $(seq 0 $((NUM_RUNS-1))); do
  sbatch <<SB
#!/bin/bash
#SBATCH --job-name=convert-benchmark${i} 
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=\$HOME/logs/omez-%A.out
#SBATCH --error=\$HOME/logs/omez-%A.err
set -euo pipefail

# activate env (direct path safest)
if [ -f "\$HOME/.pyenv/versions/benchmark/bin/activate" ]; then
  source "\$HOME/.pyenv/versions/benchmark/bin/activate"
fi

srun -c \${SLURM_CPUS_PER_TASK} \\
  python "${BENCH_PY}" \\
    --data-source "${DATA_SOURCE}" \\
    --run-index ${i} \\
    --out-root "${OUT_ROOT}"
SB
done
