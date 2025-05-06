#!/usr/bin/env bash

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate cdformer

EXP_DIR=exps/uodd
BASE_TRAIN_DIR=${EXP_DIR}/epoch20
mkdir exps
mkdir ${EXP_DIR}
mkdir ${BASE_TRAIN_DIR}

fewshot_seed=01
num_shot=10
epoch=
save_every_epoch=
lr_drop1=250
lr_drop2=450
lr=5e-5
lr_backbone=5e-6
FS_FT_DIR=${BASE_TRAIN_DIR}/seed${fewshot_seed}_${num_shot}shot_01
mkdir ${FS_FT_DIR}

python -u main.py \
    --dataset_file dataset2 \
    --backbone dinov2 \
    --num_feature_levels 1 \
    --enc_layers 6 \
    --dec_layers 6 \
    --hidden_dim 256 \
    --num_queries 300 \
    --batch_size 2 \
    --lr ${lr} \
    --lr_backbone ${lr_backbone} \
    --resume exps/dino_coco_80_size_vitl/base_train/checkpoint0049.pth \
    --fewshot_finetune \
    --fewshot_seed ${fewshot_seed} \
    --num_shots ${num_shot} \
    --epoch ${epoch} \
    --lr_drop_milestones ${lr_drop1} ${lr_drop2} \
    --warmup_epochs 50 \
    --save_every_epoch ${save_every_epoch} \
    --eval_every_epoch  \
    --output_dir ${FS_FT_DIR} \
    --category_codes_cls_loss \
2>&1 | tee ${FS_FT_DIR}/log.txt
