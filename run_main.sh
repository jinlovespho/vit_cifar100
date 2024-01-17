#!/bin/bash

# 1. Hyperparameters
ATTN_METHOD="vanilla"       # vanilla, dim_wise, token_wise 
BATCH_SIZE=128
LR=5e-4
HEAD=12
NUM_LAYERS=7
HIDDEN=384
MLP_HIDDEN=384

# 2. Proj & Exp Info
COMET_PROJ_NAME="ViT_cifar100_20240108"
EXP_NAME="LOCAL_authors_attenion_${ATTN_METHOD}"

# 3. run
CUDA_VISIBLE_DEVICES=0 python main.py   --dataset c100 \
                                        --num-classes 100 \
                                        --patch 8 \
                                        --batch-size ${BATCH_SIZE} \
                                        --eval-batch-size 1024 \
                                        --lr ${LR} \
                                        --max-epochs 200 \
                                        --dropout 0 \
                                        --head ${HEAD} \
                                        --num-layers ${NUM_LAYERS} \
                                        --hidden ${HIDDEN} \
                                        --mlp-hidden ${MLP_HIDDEN} \
                                        --project-name ${COMET_PROJ_NAME} \
                                        --experiment-memo ${EXP_NAME} \
                                        --attention_method ${ATTN_METHOD}\
                                        --api-key upJRJyzbQWeOazI7HlvvikhpG

echo "finished one experiment"