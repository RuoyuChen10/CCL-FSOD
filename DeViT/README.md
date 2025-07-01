# 🐳 DeViT backbone

## Notation

**Note1:** Initial upload, not fully checked

**Note2:** The DeViT used in this paper is derived from the v2 version of the paper on arxiv. Because its final version had not been released at that time, there may be some differences with the final code version of DeViT.

**Note3:** The main change is in this file: [https://github.com/RuoyuChen10/CCL-FSOD/blob/main/DeViT/detectron2/modeling/meta_arch/devit_ours.py](https://github.com/RuoyuChen10/CCL-FSOD/blob/main/DeViT/detectron2/modeling/meta_arch/devit_ours.py).

## 1. Early preparation

```
ln -s ./configs ./detectron2/model_zoo
```

DeViT init model checkpoints are stored in https://drive.google.com/drive/folders/1b3anUR2Gloh7XpvevPCuxWeYCvQJ7Pfz.

You need download:

```
|--weights
└──|--initial
   └──|--background
      |--few-shot
      |--open-vocabulary
```

## 2. Prepare datasets

Download the official dataset to this folder

```
|--datasets
    └──|--coco
       |--cocosplit
       |--fsod
       |  └──|--annotations
       |     |--part_1
       |     |--part_2
       |--FSVOD
       |--voc_prototype
       |--vocsplit_json
       |--VOC2007
       |--VOC2012
```

## 3. RUN

**PASCAL VOC:**

Base training on split 1:

```shell
CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --num-gpus 2  \
    --config-file configs/voc/base1.yaml  \
    MODEL.WEIGHTS  weights/initial/few-shot/vitl+rpn.pth \
    DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
    OUTPUT_DIR output/train3/voc-base1-vitl/
```

Base eval (DeViT baseline results):
```shell
for shot in 1 2 3 5 10  # if final, 10 -> 1 2 3 5 10
do
    CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --num-gpus 2 --eval-only \
        --config-file configs/voc/split1/${shot}shot.yaml \
        MODEL.WEIGHTS output/train/voc-base1-vitl/model_final.pth \
        DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
        OUTPUT_DIR output/eval/voc-base1-${shot}shot-vitl/
done
```

Finetuning with our method (DeViT w/ ours):

```shell
for split in 1
do
    for shot in 1 2 3 5 10   # if final, 10 -> 1 2 3 5 10
    do
        CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --num-gpus 2 \
            --config-file configs/voc/split${split}/${shot}shot_knowl.yaml \
            MODEL.WEIGHTS output/train/voc-base${split}-vitl/model_final.pth \
            DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
            OUTPUT_DIR output/train/voc-split${split}-ft-${shot}shot-vitl/
    done
done
```

Then all the fine-tuned weights are evaluated and the one with the largest nAP50 is selected. Note that the evaluation time is indeed very long, which is an inherent property of the DeViT baseline.
