# Copy to configs/local.sh, then edit DATA_ROOT and any scheduler settings.
# The data directory must be outside the source checkout.
DATA_ROOT="/absolute/path/to/inchworm-data"
JANUS_PYTHON="$DATA_ROOT/envs/janus/bin/python"
LLAMA_PYTHON="$DATA_ROOT/envs/llamagen/bin/python"
GENEVAL_PYTHON="$DATA_ROOT/envs/geneval/bin/python"

# Ordinary server: scripts/run.sh does not need Slurm or any account.
# Optional Slurm settings: set ACCOUNT and PARTITION for your own cluster.
ACCOUNT=""
PARTITION=""
QOS=""
GPU_REQUEST="gpu:1"
CPUS_PER_TASK=4
MEMORY="64G"
WALLTIME="02:00:00"
SHARDS=32
CONCURRENT=4
