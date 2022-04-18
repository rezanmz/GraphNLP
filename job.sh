#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=4
#SBATCH --mem=32GB
#SBATCH --time=0-05:00:003
#SBATCH --signal=SIGUSR1@90

module load gcc/9.3.0 arrow python/3.9.6
cd path/to/project/directory
source env/bin/activate

export NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.

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
    --huggingface-cache-dir path/to/datasets/cache/dir \
    --num-nodes 2
# --edge-construction-hidden-dims \
# --feature-construction-hidden-dims \
# --gcn-hidden-dims \