ATTN_METHOD="dim_wise"

CUDA_VISIBLE_DEVICES=0 python main.py   --dataset c100 \
                                        --num-classes 100 \
                                        --patch 8 \
                                        --batch-size 128 \
                                        --eval-batch-size 1024 \
                                        --lr 5e-4 \
                                        --max-epochs 200 \
                                        --dropout 0 \
                                        --head 12 \
                                        --num-layers 7 \
                                        --hidden 384 \
                                        --mlp-hidden 384 \
                                        --project-name ViT_cifar100_20240108 \
                                        --experiment-memo authors_attenion_${ATTN_METHOD} \
                                        --attention-method ${ATTN_METHOD}\
                                        --api-key upJRJyzbQWeOazI7HlvvikhpG
