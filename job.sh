#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=0-05:00:00

module load gcc/9.3.0 arrow python/3.9.6
cd path/to/project/directory
source env/bin/activate

export NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.

# Offline Huggingface transformers and datasets
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Debugging stuff
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun python src/main.py \
    --initial-embedding-model distilbert-base-uncased \
    --teacher-model bert-base-uncased \
    --batch-size 32 \
    --num-workers 4 \
    --datasets glue \
    --num-feats 768 \
    --feature-construction-output-dim 512 \
    --gcn-output-dim 768 \
    --max-epochs 100 \
    --wandb-project graph-nlp \
    --offline \
    --huggingface-cache-dir ./huggingface-cache \
    --num-nodes 2
    --checkpoints-dir ./checkpoints \
    --checkpoint-save-top-k -1
# --edge-construction-hidden-dims \
# --feature-construction-hidden-dims \
# --gcn-hidden-dims \