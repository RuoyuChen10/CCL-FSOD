SAVEDIR=checkpoints/fsod/
IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl                         
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth

# python3 train_net.py --num-gpus 2 --config-file configs/fsod/base.yaml \
#     --dist-url tcp://127.0.0.1:29000  \
#     --opts MODEL.WEIGHTS ImageNetPretrained/model_final_298dad.pkl \
#            OUTPUT_DIR ${SAVEDIR}/base_coco_pretrain

# CUDA_VISIBLE_DEVICES=1 python3 tools/model_surgery_defrcn.py --dataset fsod --method remove    \
#     --src-path checkpoints/fsod/base-single/model_final.pth    \
#     --save-dir checkpoints/fsod/base-single/
BASE_WEIGHT=checkpoints/fsod/base-single/model_reset_remove.pth

CONFIG_PATH=configs/fsod/defrcn_5shot.yaml

CUDA_VISIBLE_DEVICES=0,1 python3 train_net.py --num-gpus 2 --config-file ${CONFIG_PATH} --dist-url tcp://127.0.0.1:29000 \
    --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR checkpoints/fsod/5shot_seed10

# python3 train_net.py --num-gpus 1 --config-file ${CONFIG_PATH} --eval-only --dist-url tcp://127.0.0.1:29000 \
#     --opts TEST.PCB_ENABLE False MODEL.WEIGHTS checkpoints/fsod/5shot/model_final.pth