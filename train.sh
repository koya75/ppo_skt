#!/bin/bash

model=skt  # mask_double , sketch_transformer_patch, sketch_transformer_comv, ppo
robot=1

# Start tuning hyperparameters
python -m torch.distributed.launch \
    --nproc_per_node=2 --master_addr="localhost" --master_port=1234 train.py \
    --outdir results/hoge/ppo_hoge \
    --model ${model} \
    --epochs 10 \
    --gamma 0.99 \
    --step_offset 0 \
    --lambd 0.995 \
    --lr 0.0002 \
    --max-grad-norm 40 \
    --step_offset 0 \
    --num-envs ${robot} \
    --eval-n-runs ${robot} \
    --update-batch-interval 1 \
    --num-items 2 \
    --item-names item21 item10 \
    --isaacgym-assets-dir /opt/isaacgym/assets \
    --item-urdf-dir ./urdf \
    --steps 100000000 \
    --eval-batch-interval 20 \
    --max-step 10 \
    #--render

    #--output-sensor-images-dir /camera/ \]
    #_${model}