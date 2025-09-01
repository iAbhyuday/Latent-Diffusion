#!/bin/bash
# run_train.sh

# Set the custom process name using exec -a
source /mnt/sda/ab/envs/virtualenvs/lightning/bin/activate
exec -a "DoNotKill: abhyuday" python train_coco.py "$@"
