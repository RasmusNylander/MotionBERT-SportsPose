#!/bin/sh

### –- specify queue --
#BSUB -q gpua100

### -- set the job Name --
#BSUB -J 09

### -- ask for number of cores (default: 1) --
#BSUB -n 8

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00


# reques system-memory
#BSUB -R "rusage[mem=3GB]"

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --


# -- end of LSF options --
/zhome/0c/6/109332/miniconda3/envs/motionbert/bin/python /zhome/0c/6/109332/Projects/MotionBERT/train_sportspose.py --config /zhome/0c/6/109332/Projects/MotionBERT/sportspose/configs/con/09.yaml
