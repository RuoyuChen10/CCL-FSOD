SPLIT=1
SHOT=2
GPU_NUMS=2
SAVE_DIR=checkpoints/voc/CCL/split${SPLIT}-${SHOT}shot/


python -m tools.train_net \
    --num-gpus ${GPU_NUMS} \
    --config-file configs/PascalVOC-detection/CSCL_R_101_FPN_ft_all${SPLIT}_${SHOT}shot.yaml \
    --opts \
    MODEL.WEIGHTS checkpoints/voc/faster_rcnn${SPLIT}/model_reset_surgery.pth \
    OUTPUT_DIR ${SAVE_DIR} \

# python -m tools.test_net \
#     --num-gpus ${GPU_NUMS} \
#     --config-file configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_2shot.yaml --eval-all --opts OUTPUT_DIR checkpoints/voc/faster_rcnn-split2/bank_cluster_1-vision-split2-2

