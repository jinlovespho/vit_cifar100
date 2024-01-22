#!/bin/bash

batch_sizes=(128 256)
learning_rates=(1e-3)
heads=(2)
num_layers=(8)
hiddens=(384)
mlp_hiddens=(96)

# 1. Hyperparameters
ATTN_METHOD="vanilla"       # vanilla, dim_wise, token_wise 


# 2. Proj & Exp Info
COMET_PROJ_NAME="ViT_cifar100_20240122"


# 3. run
CUDA_GPU=0

for bs in ${batch_sizes[@]}
do
    for lr in ${learning_rates[@]}
    do
        for head in ${heads[@]}
        do
            for num_layer in ${num_layers[@]}
            do
                for hidden in ${hiddens[@]}
                do
                    for mlp_hidden in ${mlp_hiddens[@]}
                    do
                        EXP_NAME="LOCAL${CUDA_GPU}_attn${ATTN_METHOD}_bs${bs}_lr${lr}_head${head}_numlayers${num_layer}_hidden${hidden}_mlphidden${mlp_hidden}"

                        echo ${EXP_NAME} in progress.. 

                        CUDA_VISIBLE_DEVICES=${CUDA_GPU} python main.py --dataset c100 \
                                                                        --num-classes 100 \
                                                                        --patch 8 \
                                                                        --batch-size ${bs} \
                                                                        --eval-batch-size 1024 \
                                                                        --lr ${lr} \
                                                                        --max-epochs 200 \
                                                                        --dropout 0 \
                                                                        --head ${head} \
                                                                        --num-layers ${num_layer} \
                                                                        --hidden ${hidden} \
                                                                        --mlp-hidden ${mlp_hidden} \
                                                                        --project-name ${COMET_PROJ_NAME} \
                                                                        --experiment-memo ${EXP_NAME} \
                                                                        --attention_method ${ATTN_METHOD}\
                                                                        --api-key upJRJyzbQWeOazI7HlvvikhpG
                        sleep 1m

                    done
                done
            done
        done
    done
done
