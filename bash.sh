#!/bin/bash
# Submission script for Lyra
#SBATCH --job-name=Plots
#SBATCH --time=4-01:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=1000 # megabytes
#SBATCH --partition=batch
#
#SBATCH --mail-user=antonino.scurria@ulb.be
#SBATCH --mail-type=ALL

module purge
module load releases/2023a
module load Python/3.11.3-GCCcore-12.3.0

# Activate your virtual environment (must be absolute path!)
source /home/ulb/liq/ascurria/.venv/bin/activate


export PYTHONPATH=$PYTHONPATH:/home/ulb/liq/ascurria/EP---Up-to-date-code/

echo "Job start at $(date)"

python /home/ulb/liq/ascurria/feedshit/script_n92.5.py \
  --device cuda \
  --outdir runs/seed0 \
  --seed 0 \
  --use_aug \
  --do_dynamics --do_gradcheck --do_train \
  --etas 0.1 0.25 0.5 0.75 1.0 \
  --dyn_K 1000 --dyn_tol 1e-6 \
  --gradcheck_K 1000 --gradcheck_tol 1e-6 \
  --train_steps 3000 --eval_every 400 \
  --xz_K 1500 --xz_tol 1e-6 \
  --grad_cos_every 200


echo "Job end at $(date)"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8