#!/usr/bin/env bash

EXP_NAME=$1
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl                          
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
SPLIT_ID=1


# python3 train_net.py --num-gpus 2 --config-file configs/voc/base${SPLIT_ID}.yaml     \
#     --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                                   \
#            OUTPUT_DIR ${SAVE_DIR}/base${SPLIT_ID}


# python3 tools/model_surgery.py --dataset voc --method randinit                                \
#     --src-path ${SAVE_DIR}/base${SPLIT_ID}/model_final.pth                    \
#     --save-dir ${SAVE_DIR}/base${SPLIT_ID} 

# BASE_WEIGHT=${SAVE_DIR}/base${SPLIT_ID}/model_reset_surgery.pth


for seed in 0 
do
    for shot in 3
    do
        # python3 tools/create_config.py --dataset voc --config_root configs/voc               \
        #     --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
        CONFIG_PATH=configs/voc/mfdc_gfsod_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
        # OUTPUT_DIR=${SAVE_DIR}/mfdc_gfsod_novel${SPLIT_ID}/tfa-like/${shot}shot_seed${seed}
        # python3 train_net.py --num-gpus 2 --config-file ${CONFIG_PATH}                            \
        #     --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                     \
        #            TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}
        python3 train_net.py --num-gpus 2 --config-file ${CONFIG_PATH} --dist-url tcp://127.0.0.1:29001
        # rm ${CONFIG_PATH}
        # rm ${OUTPUT_DIR}/model_final.pth
    done
done
