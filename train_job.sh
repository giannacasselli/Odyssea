#!/bin/bash
#SBATCH --job-name=sds_yolo_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --cpus-per-task=8        # Request 8 CPU cores
#SBATCH --mem=32G                # Request 32GB of RAM
#SBATCH --time=12:00:00          # Bumping to 12h: 9k images + validation takes time
#SBATCH --account=def-areibi 
#SBATCH --output=slurm-%j.out    # Saves logs to a file named after the job ID

# 1. Load Modules
module purge
module load StdEnv/2023 python/3.10 cuda/12.2 opencv/4.9.0

# 2. Activate Environment
source ~/odyssea_env/bin/activate

# 3. CRUCIAL: Block the 'Ghost' Python 3.11/NumPy conflict
export PYTHONNOUSERSITE=1

# 4. Run Training
# We use the python call to ensure it uses your env's packages exactly.
# Changed to yolov8n.pt as you mentioned earlier.
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.train(data='sds_data.yaml', epochs=100, imgsz=640, device=0, workers=8, batch=16)"