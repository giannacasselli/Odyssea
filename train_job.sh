#!/bin/bash
#SBATCH --job-name=sds_yolo_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --cpus-per-task=8        # Request 8 CPU cores
#SBATCH --mem=32G                # Request 32GB of RAM
#SBATCH --time=04:00:00          # Limit to 4 hours (adjust as needed)
#SBATCH --account=def-areibi 

# 1. Load Modules
module purge
module load StdEnv/2023 python/3.10 cuda/12.2 opencv/4.9.0

# 2. Activate Environment
source ~/odyssea_env/bin/activate

# 3. Run Training
yolo train model=yolo11n.pt data=sds_data.yaml epochs=50 imgsz=640 device=0